import torch
import torch.nn as nn
from .bert import BertEncoder
from .base_model import base_encoder
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from dataclasses import dataclass
from torch import Tensor, device
from typing import Optional, Tuple
import math
import logging
from typing import List

from torch import nn



def code2real(code,sys_range=(-1,1),digit_precision=-2):
    code = code.float()
    return code * 10 ** digit_precision + sys_range[0]


def real2code(real,sys_range=(-1,1),digit_precision=-2):
    output = torch.round((real - sys_range[0]) / (10 ** digit_precision))
    return output.int()

def real2code_list(real_list,sys_range=(-1,1),digit_precision=-2):
    output = [int((real - sys_range[0]) / (10 ** digit_precision)) for real in real_list]
    return output

def code2real_list(code_list,sys_range=(-1,1),digit_precision=-2):
    output = [code * 10 ** digit_precision + sys_range[0] for code in code_list]
    return output

def check_real_list(real_list, digit_precision_front,digit_precision_back):
    for real in real_list:
        str_real = str(real)
        digit_front_len = len(str_real.split('.')[0])
        digit_back_len = len(str_real.split('.')[1])
        if digit_front_len > digit_precision_front or digit_back_len > digit_precision_back:
            return False
    return True


def realconst2str(real_list,digit_precision_front=1,digit_precision_back=2):
    str_list = []
    for real in real_list:
        str_real = str(real)
        digit_front_len = len(str_real.split('.')[0])
        digit_back_len = len(str_real.split('.')[1])
        assert digit_front_len <= digit_precision_front or digit_back_len <= digit_precision_back, "real number is not in the range"
        sl = [digit_str for digit_str in str_real]
        sl.append(',')
        str_list += sl
    return str_list

def str2realconst(str_list):
    real_list = []
    str_const = ''.join(str_list)
    str_list = str_const.split(',')
    for str_real in str_list:
        real_list.append(float(str_real))
    return real_list



def multiscale_meshgrid_gen(image_size:int,
                            dim:int = 2,
                            multiscale_ratio:float = 10,
                            multiscale_channels:int = 3,):

    assert multiscale_channels % 2 == 1, "multiscale_channels must be odd number"
    assert multiscale_ratio > 1, "multiscale_ratio must be greater than 1"

    meshgrid_array_list = []

    for i in range(-(multiscale_channels-1)//2, (multiscale_channels-1)//2+1):
        margin = multiscale_ratio ** i
        mesh_vec = torch.linspace(-margin, margin, image_size)
        meshgrid_list = torch.meshgrid([mesh_vec for _ in range(dim)])
        meshgrid = torch.stack(meshgrid_list, dim=-1)
        meshgrid_array_list.append(meshgrid)

    meshgrid_array = torch.stack(meshgrid_array_list, dim = 0)
    return meshgrid_array



def multiscale_meshgrid_patch(meshgrid_array,
                              patch_size:int,
                              is_inverse:bool = False,image_size = None):

    #assert len(meshgrid_array.shape) == 5, "meshgrid_array should be 5-dim tensor"

    if len(meshgrid_array.shape) == 5:
        b,c,h,w,_ = meshgrid_array.shape
    elif len(meshgrid_array.shape) == 4:
        b,c,h,w = meshgrid_array.shape
    else:
        raise ValueError("meshgrid_array should be 4-dim or 5-dim tensor")

    if is_inverse:
        assert image_size is not None, "original_batch should be provided when is_inverse is True"

    p = patch_size
    assert h % patch_size == 0 and w % patch_size == 0, "image_size must be divisible by the patch_size"

    if len(meshgrid_array.shape) == 5:
        if not is_inverse:
            meshgrid_array = rearrange(meshgrid_array, "b c (h1 p1) (w1 p2) d -> (b h1 w1) c p1 p2 d", p1 = p, p2 = p)
        else:
            meshgrid_array = rearrange(meshgrid_array, "(b h1 w1) c p1 p2 d -> b c (h1 p1) (w1 p2) d", h1 = image_size[0]//p,w1 = image_size[1]//p,p1 = p, p2 = p)
    else:
        if not is_inverse:
            meshgrid_array = rearrange(meshgrid_array, "b c (h1 p1) (w1 p2) -> (b h1 w1) c p1 p2", p1 = p, p2 = p)
        else:
            meshgrid_array = rearrange(meshgrid_array, "(b h1 w1) c p1 p2 -> b c (h1 p1) (w1 p2)", h1 = image_size[0]//p,w1 = image_size[1]//p, p1 = p, p2 = p)

    return meshgrid_array




def tie_encoder_decoder_weights(
    encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key: str
):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logging.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            print(module_name + " is tied")
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set(
                [module_name + "/" + sub_name for sub_name in encoder_modules.keys()]
            )
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(
                        decoder_modules[decoder_name],
                        type(encoder_modules[encoder_name]),
                    ) and len(encoder_modules) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(
        decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key
    )



def get_extended_attention_mask(
    attention_mask: Tensor,
    input_shape: Tuple[int],
    device: device,
    is_decoder: bool,
    dtype = torch.float,
) -> Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.
        device: (:obj:`torch.device`):
            The device of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if is_decoder:
            batch_size, seq_length = input_shape

            seq_ids = torch.arange(seq_length, device=device)
            causal_mask = (
                seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                <= seq_ids[None, :, None]
            )
            # in case past_key_values are used we need to add a prefix ones mask to the causal mask
            # causal and attention masks must have same type with pytorch version < 1.3
            causal_mask = causal_mask.to(attention_mask.dtype)

            if causal_mask.shape[1] < attention_mask.shape[1]:
                prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                causal_mask = torch.cat(
                    [
                        torch.ones(
                            (batch_size, seq_length, prefix_seq_len),
                            device=device,
                            dtype=causal_mask.dtype,
                        ),
                        causal_mask,
                    ],
                    axis=-1,
                )

            extended_attention_mask = (
                causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            )
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                input_shape, attention_mask.shape
            )
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(
        dtype=dtype
    )  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def convert_head_mask_to_5d(head_mask, num_hidden_layers,dtype=torch.float):
    """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
    assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
    head_mask = head_mask.to(dtype=dtype)  # switch to float if need + fp16 compatibility
    return head_mask

def get_head_mask(
     head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
) -> Tensor:
    """
    Prepare the head mask if needed.

    Args:
        head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
            The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
        num_hidden_layers (`int`):
            The number of hidden layers in the model.
        is_attention_chunked: (`bool`, *optional*, defaults to `False`):
            Whether or not the attentions scores are computed by chunks or not.

    Returns:
        `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
        `[None]` for each layer.
    """
    if head_mask is not None:
        head_mask = convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked is True:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers

    return head_mask


def invert_attention_mask(encoder_attention_mask: Tensor) -> Tensor:
    """
    Invert an attention mask (e.g., switches 0. and 1.).

    Args:
        encoder_attention_mask (`torch.Tensor`): An attention mask.

    Returns:
        `torch.Tensor`: The inverted attention mask.
    """
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
    # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
    # /transformer/transformer_layers.py#L270
    # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
    # encoder_extended_attention_mask.transpose(-1, -2))
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=torch.float)  # fp16 compatibility
    encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(torch.float).min

    return encoder_extended_attention_mask