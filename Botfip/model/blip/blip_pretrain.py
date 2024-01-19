import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from ..base_model import MomentumDistilationMixin, SharedQueueMixin
from ..blip.blip_outputs import (
    BlipOutput,
    BlipSimilarity,
    BlipIntermediateOutput,
)
from ..img_enc.Vit import *
from ..seq_enc.opseq_bert import *
from ..tokenizer.opseq_tokenizer import *
from ...common.utils import *
from ...operation.opt_model import *
from ..model_utils import *



class BlipPretrain(nn.Module, SharedQueueMixin, MomentumDistilationMixin):
    def __init__(
        self,
        config,
        tokenizer,
        funcimg_encoder,
        opseq_encoder,
        opseq_decoder,
        train_parameters_key = 'botfip_train_parameters',

    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.config = config

        queue_size = config[train_parameters_key].queue_size
        alpha = config[train_parameters_key].alpha
        embed_dim = config[train_parameters_key].embed_dim
        momentum= config[train_parameters_key].momentum
        tie_enc_dec_weights = config[train_parameters_key].tie_enc_dec_weights

        if tie_enc_dec_weights:
            tie_encoder_decoder_weights(
                encoder=opseq_encoder.bert,
                decoder=opseq_decoder.encoder.bert,
                base_model_prefix="",
                skip_key="/attention",
            )

        self.funcimg_encoder = funcimg_encoder

        self.opseq_encoder = opseq_encoder
        self.opseq_decoder = opseq_decoder

        if config[train_parameters_key].load_path  is not None:
            print("loading model from ", config.load_path)
            self.load_models(config.load_path)

        # creating projection layers for ITC
        opseq_width = opseq_encoder.config.hidden_size
        funcimg_width = funcimg_encoder.config.vision_width

        self.funcimg_proj = nn.Linear(funcimg_width, embed_dim)
        self.opseq_proj = nn.Linear(opseq_width, embed_dim)

        self.itm_head = nn.Linear(opseq_width, 2)

        # create the momentum encoder
        self.funcimg_encoder_m = deepcopy(self.funcimg_encoder)
        self.opseq_encoder_m = deepcopy(self.opseq_encoder)

        self.funcimg_proj_m = deepcopy(self.funcimg_proj)
        self.opseq_proj_m = deepcopy(self.opseq_proj)

        self.model_pairs = [
            [self.funcimg_encoder, self.funcimg_encoder_m],
            [self.opseq_encoder, self.opseq_encoder_m],
            [self.funcimg_proj, self.funcimg_proj_m],
            [self.opseq_proj, self.opseq_proj_m],
        ]
        self.copy_params()

        # create the queue
        self.register_buffer("funcimg_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("opseq_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.funcimg_queue = nn.functional.normalize(self.funcimg_queue, dim=0)
        self.opseq_queue = nn.functional.normalize(self.opseq_queue, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.alpha = alpha

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))


    def funcimg_forward(self, funcimg_input):
        funcimg_output = self.funcimg_encoder(funcimg_input)
        funcimg_embeds = funcimg_output.last_hidden_state
        funcimg_mlp_output = funcimg_output.mlp_output
        funcimg_atts = torch.ones(funcimg_embeds.size()[:-1], dtype=torch.long).to(
            funcimg_input.device
        )
        funcimg_feat = F.normalize(self.funcimg_proj(funcimg_embeds[:, 0, :]), dim=-1)
        return funcimg_embeds,funcimg_feat,funcimg_atts,funcimg_mlp_output


    def opseq_generate(self,
                       funcimg,
                        device = 'cuda',
                        if_compute_constant=True,
                        ):

        self.eval()
        bs = funcimg.size(0)

        funcimg = funcimg.to(device)
        funcimg_embeds, funcimg_feat, funcimg_atts, funcimg_mlp_output = self.funcimg_forward(funcimg)
        seq_input_ids, seq_input_constants, seq_input_mask = self.tokenizer.generate_empty_token(start_token='bos')

        seq_input_ids = repeat(seq_input_ids, '1 ... -> b ...', b=bs)
        seq_input_constants = repeat(seq_input_constants, '1 ... -> b ...', b=bs)
        seq_input_mask = repeat(seq_input_mask, '1 ... -> b ...', b=bs)

        pad_ids = self.tokenizer.getid('pad')
        stop_ids = self.tokenizer.getid('eos')


        with torch.no_grad():
            output_opseq, output_constant = self.opseq_decoder.generate_from_funcimg(seq_input_ids,
                                                                                     seq_input_constants,
                                                                                    seq_input_mask,
                                                                                    funcimg_embeds,
                                                                                    funcimg_atts,
                                                                                    if_compute_constant=if_compute_constant,
                                                                                    stop_ids=stop_ids,
                                                                                    device = device,)

        return output_opseq, output_constant



    def lm_loss_cal(self,opseq_token,op_mask,op_constants,funcimg_embeds,funcimg_atts):
        # LM

        decoder_input_opseq_token = opseq_token.clone()
        op_constants = op_constants.clone()
        decoder_input_opseq_token[:, 0] = self.tokenizer.token2ind(self.tokenizer.bos_token)
        decoder_opseq_token_targets = decoder_input_opseq_token.masked_fill(
            decoder_input_opseq_token == self.tokenizer.token2ind(self.tokenizer.pad_token), -100
        )

        min_constants_range = self.config.operation_tree_config.constants_range[0]
        decoder_input_opseq_constants = torch.ones_like(op_constants).to(op_constants.device)
        decoder_input_opseq_constants[:, 0] = self.tokenizer.token2ind(self.tokenizer.bos_token)
        decoder_input_opseq_constants[:, 1:] = self.tokenizer.token2ind(
            self.tokenizer.pad_token)

        decoder_input_op_mask = op_mask.clone()
        seq_len = decoder_input_opseq_token.size(-1)
        decoder_input_op_mask[:, seq_len:] = 0

        constant_mask = op_mask.clone()
        constant_mask = constant_mask[:, seq_len:].to(torch.bool)

        prediction_scores, pred_const_array = self.opseq_decoder(
            decoder_input_opseq_token,
            decoder_input_opseq_constants,
            seq_input_mask=decoder_input_op_mask,
            encoder_hidden_states=funcimg_embeds,
            encoder_attention_mask=funcimg_atts,
            mode = 'multimodal',
        )


        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        seq_labels = decoder_opseq_token_targets[:, 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)

        loss_lm = loss_fct(
            shifted_prediction_scores.view(-1, shifted_prediction_scores.size(-1)),
            seq_labels.view(-1).long(),
        )




        return loss_lm, prediction_scores, pred_const_array, decoder_opseq_token_targets


    def const_update(self,
                      config_yaml,
                      opseq,
                      target_funcimg,
                      input_meshgrid,
                      op_constant_batch=None,
                      device='cuda',
                      optimizer='lbfgs',
                      lr=1,
                      scheduler_steps=(2000, 5000, 10000, 15000),
                      gamma=0.1,
                      print_epochs=100,
                      max_epochs=20000,
                      threshold=1e-3,
                      ):

        b_opt = opt_batch_model.build(config_yaml,opseq,op_constant_batch=op_constant_batch).to(device).float()
        input_meshgrid = input_meshgrid.float().to(device)
        target_funcimg = target_funcimg.float().to(device)
        if optimizer == 'adamw':
            optimizer = torch.optim.AdamW(b_opt.parameters(), lr=lr)
        elif optimizer == 'lbfgs':
            optimizer = torch.optim.LBFGS(b_opt.parameters(), lr=lr)
        else:
            raise NotImplementedError
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_steps, gamma=gamma)
        last_loss = 0
        for epoch in range(max_epochs):
            def closure():
                optimizer.zero_grad()
                pred_funcimg = b_opt(input_meshgrid)
                loss = F.mse_loss(pred_funcimg, target_funcimg)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            if torch.isinf(loss).sum() > 0 or torch.isnan(loss).sum() > 0:
                raise ValueError('loss is inf or nan')
            scheduler.step()
            if epoch % print_epochs == 0:
                print(f'epoch {epoch}: loss {loss.item()}')
            if epoch==0:
                last_loss = loss.item()
            elif (loss.item() - last_loss) < threshold:
                print(f'epoch {epoch}: loss {loss.item()}')
                break
            else:
                last_loss = loss.item()
        constants_array = [consts.detach().cpu().numpy() for consts in b_opt.constants]
        pred_funcimg = b_opt(input_meshgrid).detach().cpu().numpy()
        return constants_array,pred_funcimg,loss.item()




    def save_models(self,save_dir):
        save_path = os.path.join(self.config.botfip_train_parameters.model_save_path,save_dir)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        funcimg_encoder_save_path = os.path.join(save_path,'funcimg_encoder.pth')
        opseq_encoder_save_path = os.path.join(save_path,'opseq_encoder.pth')
        opseq_decoder_save_path = os.path.join(save_path,'opseq_decoder.pth')
        total_save_path = os.path.join(save_path,'total.pth')

        torch.save(self.funcimg_encoder.state_dict(),funcimg_encoder_save_path)
        torch.save(self.opseq_encoder.state_dict(),opseq_encoder_save_path)
        torch.save(self.opseq_decoder.state_dict(),opseq_decoder_save_path)
        torch.save(self.state_dict(),total_save_path)


    def load_models(self,save_path):
        funcimg_encoder_save_path = os.path.join(save_path,'funcimg_encoder.pth')
        opseq_encoder_save_path = os.path.join(save_path,'opseq_encoder.pth')
        opseq_decoder_save_path = os.path.join(save_path,'opseq_decoder.pth')

        self.funcimg_encoder.load_state_dict(torch.load(funcimg_encoder_save_path))
        self.opseq_encoder.load_state_dict(torch.load(opseq_encoder_save_path))
        self.opseq_decoder.load_state_dict(torch.load(opseq_decoder_save_path))

    def forward(self,
                samples,
                cal_itc_itm_loss=True,
                cal_lm_loss=False,
                encoder_freeze=False,):

        funcimg_input = samples["funcimg"]
        opseq_input, op_constants_input = samples["opseq"]

        if (torch.isnan(funcimg_input) | torch.isinf(funcimg_input)).any():
            print('funcimg_input nan of inf')
            print(funcimg_input)
            raise ValueError

        alpha = self.alpha * self._rampup_factor(
            epoch=samples["epoch"],
            iters=samples["iters"],
            num_iters_per_epoch=samples["num_iters_per_epoch"],
        )

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        # image embeddings and features
        funcimg_output = self.funcimg_encoder(funcimg_input)
        funcimg_embeds = funcimg_output.last_hidden_state
        funcimg_mlp_output = funcimg_output.mlp_output
        funcimg_atts = torch.ones(funcimg_embeds.size()[:-1], dtype=torch.long).to(
            funcimg_input.device
        )
        funcimg_feat = F.normalize(self.funcimg_proj(funcimg_embeds[:, 0, :]), dim=-1)

        opseq_token, op_constants, op_mask = self.tokenizer.tokenize(opseq_input, const_array=op_constants_input,device=funcimg_input.device)

        opseq_token = opseq_token.to(funcimg_input.device)
        op_constants = op_constants.to(funcimg_input.device)
        op_mask = op_mask.to(funcimg_input.device)

        # opseq embeddings and features
        opseq_output, _ = self.opseq_encoder(opseq_token, op_constants, op_mask, is_only_hidden=False,mode="seq")

        opseq_embeds = opseq_output.last_hidden_state
        opseq_feat = F.normalize(self.opseq_proj(opseq_embeds[:, 0, :]), dim=-1)

        if encoder_freeze:
            funcimg_feat = funcimg_feat.detach()
            opseq_feat = opseq_feat.detach()
            opseq_embeds = opseq_embeds.detach()
            funcimg_embeds = funcimg_embeds.detach()
            funcimg_mlp_output = funcimg_mlp_output.detach()

        loss = 0
        if cal_itc_itm_loss:
            # get momentum features
            with torch.no_grad():
                self._momentum_update()
                funcimg_embeds_m = self.funcimg_encoder_m(funcimg_input).last_hidden_state
                funcimg_feat_m = F.normalize(
                    self.funcimg_proj_m(funcimg_embeds_m[:, 0, :]), dim=-1
                )
                funcimg_feat_all = torch.cat(
                    [funcimg_feat_m.t(), self.funcimg_queue.clone().detach()], dim=1
                )

                opseq_output_m, _ = self.opseq_encoder_m(opseq_token, op_constants, op_mask, is_only_hidden=False,mode="seq")
                opseq_embeds_m = opseq_output_m.last_hidden_state
                opseq_feat_m = F.normalize(self.opseq_proj_m(opseq_embeds_m[:, 0, :]), dim=-1)
                opseq_feat_all = torch.cat(
                    [opseq_feat_m.t(), self.opseq_queue.clone().detach()], dim=1
                )

                sim_i2t_m = funcimg_feat_m @ opseq_feat_all / self.temp
                sim_t2i_m = opseq_feat_m @ funcimg_feat_all / self.temp

                sim_targets = torch.zeros(sim_i2t_m.size()).to(funcimg_input.device)
                sim_targets.fill_diagonal_(1)

                sim_i2t_targets = (
                        alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                )
                sim_t2i_targets = (
                        alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
                )

            sim_i2t = funcimg_feat @ opseq_feat_all / self.temp
            sim_t2i = opseq_feat @ funcimg_feat_all / self.temp

            loss_i2t = -torch.sum(
                F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1
            ).mean()
            loss_t2i = -torch.sum(
                F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1
            ).mean()

            loss_itc = (loss_i2t + loss_t2i) / 2

            self._dequeue_and_enqueue(funcimg_feat_m, opseq_feat_m)



            encoder_input_opseq_token = opseq_token.clone()
            encoder_input_masked_op_constants = op_constants.clone()

            encoder_input_opseq_token[:, 0] = self.tokenizer.token2ind(self.tokenizer.enc_token)


            encoder_input_masked_op_constants[:, 0] = self.tokenizer.token2ind(self.tokenizer.enc_token)
            encoder_input_masked_op_constants[:, 1:] = self.tokenizer.token2ind(
                self.tokenizer.pad_token)

            encoder_input_op_mask = op_mask.clone()
            seq_len = encoder_input_opseq_token.size(-1)
            encoder_input_op_mask[:, seq_len:] = 0

            bs = funcimg_input.size(0)
            output_pos, _ = self.opseq_encoder(
                encoder_input_opseq_token,
                encoder_input_masked_op_constants,
                encoder_input_op_mask,
                encoder_hidden_states=funcimg_embeds,
                encoder_attention_mask=funcimg_atts,
                is_only_hidden=False,
                mode="multimodal",
            )

            with torch.no_grad():
                same_opseq_ind = get_same_ind(encoder_input_opseq_token)
                weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + 1e-4
                # weights_t2i.fill_diagonal_(0)
                weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + 1e-4
                # weights_i2t.fill_diagonal_(0)
                weights_t2i = weights_t2i * same_opseq_ind
                weights_i2t = weights_i2t * same_opseq_ind

            # select a negative image for each opseq
            funcimg_embeds_neg = []
            for b in range(bs):
                if not (weights_t2i[b] == 0).all():
                    try:
                        neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                        funcimg_embeds_neg.append(funcimg_embeds[neg_idx])
                    except Exception as e:
                        funcimg_embeds_neg_random = torch.randn_like(funcimg_embeds[0])
                        funcimg_embeds_neg.append(funcimg_embeds_neg_random)
                else:
                    funcimg_embeds_neg_random = torch.randn_like(funcimg_embeds[0])
                    funcimg_embeds_neg.append(funcimg_embeds_neg_random)

            funcimg_embeds_neg = torch.stack(funcimg_embeds_neg, dim=0)

            # select a negative opseq for each image
            opseq_token_neg = []
            op_constants_neg = []
            op_mask_neg = []
            for b in range(bs):
                if not (weights_i2t[b] == 0).all():
                    try:
                        neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                        opseq_token_neg.append(encoder_input_opseq_token[neg_idx])
                        op_constants_neg.append(encoder_input_masked_op_constants[neg_idx])
                        op_mask_neg.append(op_mask[neg_idx])
                    except Exception as e:
                        raise e

                else:
                    encoder_input_opseq_token_random, _ = self.tokenizer.random_generate_token(
                        encoder_input_opseq_token[0].shape)
                    encoder_input_masked_op_constants_random = encoder_input_masked_op_constants[0].clone()
                    encoder_input_op_mask = torch.ones_like(op_mask[0], dtype=torch.int)
                    encoder_input_op_mask[seq_len:] = 0
                    opseq_token_neg.append(encoder_input_opseq_token_random)
                    op_constants_neg.append(encoder_input_masked_op_constants_random)
                    op_mask_neg.append(encoder_input_op_mask)

            opseq_token_neg = torch.stack(opseq_token_neg, dim=0)
            op_constants_neg = torch.stack(op_constants_neg, dim=0)
            op_mask_neg = torch.stack(op_mask_neg, dim=0)

            opseq_token_all = torch.cat([encoder_input_opseq_token, opseq_token_neg], dim=0)
            op_constants_all = torch.cat([encoder_input_masked_op_constants, op_constants_neg], dim=0)
            op_mask_all = torch.cat([op_mask, op_mask_neg], dim=0)

            funcimg_embeds_all = torch.cat([funcimg_embeds, funcimg_embeds_neg], dim=0)
            funcimg_atts_all = torch.cat([funcimg_atts, funcimg_atts], dim=0)

            output_neg, _ = self.opseq_encoder(
                opseq_token_all,
                op_constants_all,
                op_mask_all,
                encoder_hidden_states=funcimg_embeds_all,
                encoder_attention_mask=funcimg_atts_all,
                is_only_hidden=False,
                mode="multimodal",
            )

            vl_embeddings = torch.cat(
                [
                    output_pos.last_hidden_state[:, 0, :],
                    output_neg.last_hidden_state[:, 0, :],
                ],
                dim=0,
            )

            itm_logits = self.itm_head(vl_embeddings)

            itm_labels = torch.cat(
                [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                dim=0,
            ).to(funcimg_input.device)

            loss_itm = F.cross_entropy(itm_logits, itm_labels)
            #print('loss_itm:', loss_itm)
            loss = loss_itc + loss_itm
        else:
            loss_itc = None
            loss_itm = None
            itm_logits = None
            itm_labels = None
            sim_i2t = None
            sim_t2i = None
            output_pos = None
            output_neg = None

        if cal_lm_loss:
            loss_lm, prediction_scores, pred_const_array, decoder_opseq_token_targets = self.lm_loss_cal(
                opseq_token, op_mask, op_constants, funcimg_embeds, funcimg_atts)
            loss = loss + loss_lm
        else:
            loss_lm = None
            prediction_scores = None
            pred_const_array = None
            decoder_opseq_token_targets = None


        '''
        if cal_itc_itm_loss:
            sims = BlipSimilarity(
                sim_i2t=sim_i2t,
                sim_t2i=sim_t2i,
                sim_i2t_m=sim_i2t_m,
                sim_t2i_m=sim_t2i_m,
                sim_i2t_targets=sim_i2t_targets,
                sim_t2i_targets=sim_t2i_targets,
            )
            intermediate_output = BlipIntermediateOutput(
                funcimg_embeds=funcimg_embeds,
                funcimg_mlpoutput=funcimg_mlp_output,
                opseq_embeds=opseq_embeds,
                funcimg_embeds_m=funcimg_embeds_m,
                opseq_embeds_m=opseq_embeds_m,
                encoder_output=output_pos,
                encoder_output_neg=output_neg,
                itm_logits=itm_logits,
                itm_labels=itm_labels,
                decoder_output_seq_logits=prediction_scores,
                decoder_output_constants=pred_const_array,
                decoder_seq_labels=decoder_opseq_token_targets,
            )
        else:

        '''
        sims = None
        intermediate_output = None

        return BlipOutput(
            loss=loss,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
            sims=sims,
            intermediate_output=intermediate_output
        )


    def reset_queue_ptr(self):
        self.queue_ptr = torch.zeros(1, dtype=torch.long)

    @classmethod
    def from_config(cls, config,train_parameters_key):
        tokenizer = op_tokenizer.from_config(config)
        funcimg_encoder = ViT(config.funcimg_encoder_config)
        opseq_encoder = OpSeq_Encoder(config, tokenizer.vocab_size)
        opseq_decoder = OpSeq_Decoder(config, tokenizer.vocab_size)

        return cls(config, tokenizer, funcimg_encoder,opseq_encoder,opseq_decoder,train_parameters_key)






