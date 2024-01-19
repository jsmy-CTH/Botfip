import yaml
from Botfip.model.bert import *
from Botfip.model.model_utils import *
from dataclasses import dataclass




class OpSeq_Embedding(nn.Module):
    def __init__(self,
                 config,
                 vocab_size:int,):

        super().__init__()

        self.vocab_size = vocab_size
        self.max_constants_num = config.max_constants_num
        self.embed_dim = config.hidden_size
        self.max_seq_length = config.max_seq_length
        self.op_seq_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.const_embedding = nn.Linear(1, self.embed_dim)

        self.register_buffer(
            "position_ids", torch.arange(self.max_constants_num+self.max_seq_length+2).expand((1, -1))
        )

        self.position_embeddings = nn.Embedding(self.max_constants_num+self.max_seq_length+4, self.embed_dim)

        self.LayerNorm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    @classmethod
    def from_config(cls,config,*args,**kwargs):
        opseq_config = config['opseq_tokenizer_config']
        opseq_config.update(config['bert_config'])
        opseq_config.update(config['operation_tree_config'])

        return cls(opseq_config,*args,**kwargs)

    @classmethod
    def from_config_yaml(cls,model_config,*args,**kwargs):
        with open(model_config, 'r') as file:
            m_dict = yaml.safe_load(file)
        model_config_dict= m_dict['opseq_tokenizer_config']
        model_config_dict.update(m_dict['bert_config'])

        for k,v in model_config_dict.items():
            if isinstance(v,dict):
                for k1,v1 in v.items():
                    kwargs[k1]=v1
            else:
                kwargs[k]=v
        return cls(*args,**kwargs)


    def forward(
        self,
        input_op_seq_tensor,
        input_constants_tensor,
    ):
        # op_seq_embedding
        op_seq_embeddings = self.op_seq_embedding(input_op_seq_tensor)
        # const_embedding
        const_embeddings = self.const_embedding(input_constants_tensor.unsqueeze(-1))
        # concat op_seq_embedding and const_embedding
        embeddings = torch.cat([op_seq_embeddings, const_embeddings], dim=1)

        pos_embeddings = self.position_embeddings(self.position_ids)
        embeddings += pos_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings



class OpSeq_decoder_layer(nn.Module):
    def __init__(self,
                 config,
                 vocab_size:int,
                 ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.hidden_layer = nn.Sequential(
            nn.Linear(config.hidden_size, 2*config.hidden_size),
            nn.PReLU(),
            nn.Linear(2*config.hidden_size, 4*config.hidden_size),
            nn.PReLU(),
        )
        self.hidden2seq = nn.Linear(4*config.hidden_size, vocab_size)
        self.hidden2const = nn.Linear(4*config.hidden_size, 1)

    def forward(self,hidden_states,seq_len):
        hidden_states = self.hidden_layer(hidden_states)
        logits_seq = self.hidden2seq(hidden_states[:, :seq_len])
        const_array = self.hidden2const(hidden_states[:,seq_len:])
        return logits_seq,const_array






class OpSeq_Encoder(nn.Module):
    def __init__(self,
                 config,
                 vocab_size,
                 add_pooling_layer=True,

                 ):
        super().__init__()
        self.config = config.bert_config

        self.opseq_embedding = OpSeq_Embedding.from_config(config,vocab_size=vocab_size)

        self.bert = BertEncoder(config.bert_config)

        self.pooler = BertPooler(config.bert_config) if add_pooling_layer else None

        self.model_type_map={'seq':'text','multimodal':'multimodal'}

    def forward(self,
                seq_input_ids,
                seq_input_constants,
                seq_input_mask,
                head_mask=None,
                past_key_values=None,
                use_cache=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                mode="multimodal",
                is_only_hidden = True,
                is_decoder=False,
                ):


        device = seq_input_ids.device

        if mode not in self.model_type_map.keys():
            raise ValueError("mode {} is not supported".format(mode))

        mix_embedding_tensor = self.opseq_embedding(seq_input_ids,seq_input_constants)
        assert mix_embedding_tensor.size(1) == seq_input_mask.size(1), "seq_input_mask size is not match with mix_embedding_tensor"

        batch_size = mix_embedding_tensor.size(0)
        seq_length = mix_embedding_tensor.size(1)
        input_shape = mix_embedding_tensor.size()[:-1]

        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if seq_input_mask is None:
            seq_input_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        extended_attention_mask: torch.Tensor = get_extended_attention_mask(
            seq_input_mask, input_shape, device, is_decoder
        )

        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[
                    0
                ].size()
            else:
                (
                    encoder_batch_size,
                    encoder_sequence_length,
                    _,
                ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [
                    invert_attention_mask(mask) for mask in encoder_attention_mask
                ]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = invert_attention_mask(
                    encoder_attention_mask
                )
            else:
                encoder_extended_attention_mask = invert_attention_mask(
                    encoder_attention_mask
                )
        else:
            encoder_extended_attention_mask = None

        head_mask = get_head_mask(head_mask, self.config.num_hidden_layers)


        encoder_outputs = self.bert(mix_embedding_tensor,
                                   attention_mask = extended_attention_mask,
                                   head_mask=head_mask,
                                   past_key_values=past_key_values,
                                   encoder_hidden_states=encoder_hidden_states,
                                   encoder_attention_mask=encoder_extended_attention_mask,
                                   use_cache=use_cache,
                                   mode = self.model_type_map[mode],
                                    )
        pooled_output = (
            self.pooler(encoder_outputs.last_hidden_state) if self.pooler is not None else None
        )

        if is_only_hidden:
            return encoder_outputs.last_hidden_state,pooled_output
        else:
            return encoder_outputs,pooled_output


class OpSeq_Decoder(nn.Module):
    def __init__(self,
                 config,
                 vocab_size,
                 add_pooling_layer=True
                 ):
        super().__init__()
        self.config = config

        self.encoder = OpSeq_Encoder(config,vocab_size,add_pooling_layer)
        self.decoder_layer = OpSeq_decoder_layer(config.bert_config,vocab_size)


    def forward(self,
                seq_input_ids,
                seq_input_constants,
                seq_input_mask=None,
                head_mask=None,
                past_key_values=None,
                use_cache=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                mode="multimodal",
                ):

        seq_len = seq_input_ids.size(1)
        encoder_output,_ = self.encoder(seq_input_ids,
                                        seq_input_constants,
                                        seq_input_mask=seq_input_mask,
                                        head_mask=head_mask,
                                        past_key_values=past_key_values,
                                        use_cache=use_cache,
                                        encoder_hidden_states=encoder_hidden_states,
                                        encoder_attention_mask=encoder_attention_mask,
                                        mode=mode,
                                        is_only_hidden = True,
                                        is_decoder=True,)
        logits_seq,const_array = self.decoder_layer(encoder_output,seq_len = seq_len)
        return logits_seq,const_array


    def generate_from_funcimg(self,
                              seq_input_ids,
                              seq_input_constants,
                              seq_input_mask,
                              funcimg_encoder_hidden_state,
                              funcimg_encoder_attention_mask,
                              stop_ids=None,
                              pad_ids=None,
                              max_seq_length = 30,
                              head_mask=None,
                              past_key_values=None,
                              use_cache=None,
                              if_compute_constant=False,
                              device = 'cuda'):

        self.eval()
        seq_input_ids = seq_input_ids.clone().to(device)
        seq_input_constants = seq_input_constants.clone().to(device)
        seq_input_mask = seq_input_mask.clone().to(device)
        funcimg_encoder_hidden_state = funcimg_encoder_hidden_state.clone().to(device)
        funcimg_encoder_attention_mask = funcimg_encoder_attention_mask.clone().to(device)

        start_generate_ids = min([torch.where(seq_input_mask[i]==0)[0][0].item() for i in range(seq_input_mask.shape[0])])
        if pad_ids is not None:
            seq_input_ids[:,start_generate_ids:] = pad_ids


        complete_flag = torch.zeros(seq_input_ids.size(0))
        output_seq_list = []

        for i in range(start_generate_ids ,max_seq_length+2):
            logits_seq,_ = self.forward(seq_input_ids,
                                        seq_input_constants,
                                        seq_input_mask=seq_input_mask,
                                        head_mask=head_mask,
                                        past_key_values=past_key_values,
                                        use_cache=use_cache,
                                        encoder_hidden_states=funcimg_encoder_hidden_state,
                                        encoder_attention_mask=funcimg_encoder_attention_mask,
                                        mode="multimodal",)
            next_token_logits = logits_seq[:, i-1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).type_as(seq_input_ids)

            change_batch = torch.where(complete_flag==0)[0]
            seq_input_ids[change_batch, i] = next_token[change_batch]
            seq_input_mask[change_batch, i] = 1
            if stop_ids is not None:
                flag = next_token==stop_ids
                flag = flag.detach().cpu()
                complete_flag = torch.where(flag,1,complete_flag)
                if complete_flag.sum() == seq_input_ids.size(0):
                    break

        if if_compute_constant:
            _,pred_const_array = self.forward(seq_input_ids,
                                              seq_input_constants,
                                              seq_input_mask=seq_input_mask,
                                              head_mask=head_mask,
                                              past_key_values=past_key_values,
                                              use_cache=use_cache,
                                              encoder_hidden_states=funcimg_encoder_hidden_state,
                                              encoder_attention_mask=funcimg_encoder_attention_mask,
                                              mode="multimodal",)
        else:
            pred_const_array = None

        for i in range(seq_input_ids.size(0)):
            chunk_index = torch.where(seq_input_mask[i]==0)[0][0].item()
            output_seq_list.append(seq_input_ids[i,:chunk_index].tolist())

        return output_seq_list,pred_const_array



















