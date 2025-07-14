# -*- coding: utf-8 -*-

from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer)
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb, LlamaDecoderLayer
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig
from typing import Optional, Tuple, Union, Dict

import torch
import torch.nn as nn

from einops import rearrange
import torch.nn.functional as F
from small_fla import fused_recurrent_gla

from collections import OrderedDict 

def convert_transformer_to_dsla(model, config, checkpoints, layer_to_convert): 
    """
    Replaces a specific LlamaAttention layer with DualStateLinearAttention
    and loads the corresponding weights.

    Args:
        model (nn.Module): The entire LlamaForCausalLM model.
        config (LlamaConfig): The model configuration.
        checkpoints (Dict[str, torch.Tensor]): The full state_dict loaded from the original model checkpoint.
        layer_to_convert (int): The 0-based index of the layer to convert.

    Returns:
        nn.Module: The modified model.
    """
    
    target_attention_module = None
    original_device = None

    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        raise ValueError("Model structure not as expected: does not have model.layers.")

    for li, layer in enumerate(model.model.layers):
        layer.layer_idx = li
        if hasattr(layer, 'self_attn'):
            layer.self_attn.layer_idx = li

            if li == layer_to_convert:
                target_attention_module = layer.self_attn
                original_device = target_attention_module.q_proj.weight.device
                print(f"Found target LlamaAttention at layer {li} on device {original_device}")
                break

    if target_attention_module is None:
        print(f"Warning: LlamaAttention layer at index {layer_to_convert} not found. No conversion performed.")
        return model

    print(f"Replacing layer {layer_to_convert} attention to DualStateLinearAttention...")
    new_dsla_module = DualStateLinearAttention(config)
    new_dsla_module.layer_idx = layer_to_convert
    new_dsla_module.to(original_device)
    new_dsla_module.to(torch.float16)
    model.model.layers[layer_to_convert].self_attn = new_dsla_module
    print(f"Layer {layer_to_convert} replaced. New module on device: {new_dsla_module.q_proj.weight.device}")

    filtered_state_dict_for_new_layer = OrderedDict()
    for suffix in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        weight_key = f'model.layers.{layer_to_convert}.self_attn.{suffix}.weight'
        if weight_key in checkpoints:
            filtered_state_dict_for_new_layer[weight_key] = checkpoints[weight_key].to(original_device)
        else:
            print(f"Warning: Expected weight key {weight_key} not found in checkpoint for replacement.")

    for suffix in ['gk_proj', 'gk_proj2']:
        weight_key = f'model.layers.{layer_to_convert}.self_attn.{suffix}.weight'
        bias_key = f'model.layers.{layer_to_convert}.self_attn.{suffix}.bias' 
        if weight_key in checkpoints:
            filtered_state_dict_for_new_layer[weight_key] = checkpoints[weight_key].to(original_device)
        if bias_key in checkpoints:
            filtered_state_dict_for_new_layer[bias_key] = checkpoints[bias_key].to(original_device)
    
    alpha_list_key = f'model.layers.{layer_to_convert}.self_attn.alpha_list'
    if alpha_list_key in checkpoints:
        filtered_state_dict_for_new_layer[alpha_list_key] = checkpoints[alpha_list_key].to(original_device)

    model.load_state_dict(filtered_state_dict_for_new_layer, strict=False)
    print(f"Weights for layer {layer_to_convert} (and custom DSLA parts) loaded from checkpoint.")

    return model


def convert_dsla_to_transformer(model, config, checkpoints, layer_to_convert): 
    """
    Replaces a specific DualStateLinearAttention layer with LlamaAttention
    and loads the corresponding original LlamaAttention weights.

    Args:
        model (nn.Module): The entire LlamaForCausalLM model (currently containing DSLA).
        config (LlamaConfig): The model configuration.
        checkpoints (Dict[str, torch.Tensor]): The full state_dict loaded from the original Llama model.
        layer_to_convert (int): The 0-based index of the layer to convert back.

    Returns:
        nn.Module: The modified model (with LlamaAttention restored).
    """
    
    target_dsla_module = None
    original_device = None

    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        raise ValueError("Model structure not as expected: does not have model.layers.")

    for li, layer in enumerate(model.model.layers):
        layer.layer_idx = li
        if hasattr(layer, 'self_attn'):
            layer.self_attn.layer_idx = li

            if isinstance(layer.self_attn, DualStateLinearAttention) and li == layer_to_convert:
                target_dsla_module = layer.self_attn
                original_device = target_dsla_module.q_proj.weight.device
                print(f"Found target DualStateLinearAttention at layer {li} on device {original_device}")
                break

    if target_dsla_module is None:
        print(f"Warning: DualStateLinearAttention layer at index {layer_to_convert} not found. No conversion performed.")
        return model

    print(f"Converting layer {layer_to_convert} attention back to LlamaAttention...")
    new_llama_attention_module = LlamaAttention(config, layer_idx=layer_to_convert) 
    new_llama_attention_module.to(original_device)
    new_llama_attention_module.to(torch.float16)
    model.model.layers[layer_to_convert].self_attn = new_llama_attention_module
    print(f"Layer {layer_to_convert} replaced with LlamaAttention. New module on device: {new_llama_attention_module.q_proj.weight.device}")

    filtered_state_dict_for_llama_attn = OrderedDict()
    for suffix in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        weight_key = f'model.layers.{layer_to_convert}.self_attn.{suffix}.weight'
        bias_key = f'model.layers.{layer_to_convert}.self_attn.{suffix}.bias' 
        if weight_key in checkpoints:
            filtered_state_dict_for_llama_attn[weight_key] = checkpoints[weight_key].to(original_device)
        else:
            print(f"Warning: Expected LlamaAttention weight key {weight_key} not found in original checkpoint for layer {layer_to_convert}.")
        
        if bias_key in checkpoints:
            filtered_state_dict_for_llama_attn[bias_key] = checkpoints[bias_key].to(original_device)

    model.load_state_dict(filtered_state_dict_for_llama_attn, strict=False)
    print(f"Original LlamaAttention weights for layer {layer_to_convert} loaded from checkpoint.")

    return model

def convert_transformer_to_dsla_multiple(model, config):
    # convert all layers in the 
    def change_class(model, config, dsla_list):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                model._modules[name] = change_class(module, config, dsla_list)
            if isinstance(module, LlamaAttention) and module.layer_idx in dsla_list:
                model._modules[name] = DualStateLinearAttention(config)
        return model
    dsla_list = config.dsla_list
    for li, l in enumerate(model.model.layers):
        l.layer_idx = li
        if type(l) == LlamaDecoderLayer:
            l.self_attn.layer_idx = li
    device = model.device
    model = change_class(model, config, dsla_list)
    for li, l in enumerate(model.model.layers):
        l.layer_idx = li
        if type(l) == LlamaDecoderLayer:
            l.self_attn.layer_idx = li

    return model.to(device)

class DualStateLinearAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
     
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )    
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

        self.ker_dim = config.kernel_hidden_size

        self.gk_proj = nn.Linear(self.head_dim * self.num_heads, self.ker_dim * self.num_heads, bias=True)
        nn.init.eye_(self.gk_proj.weight)
        self.gk_proj2 = nn.Linear(self.head_dim * self.num_heads, self.ker_dim * self.num_heads, bias=True)
        self.gk_proj.bias.data.fill_(0.0)
        self.gk_proj2.bias.data.fill_(0.0)
        gate_logit_normalizer = 16 
        self.gate_logit_normalizer = gate_logit_normalizer
        clamp_min: Optional[float] = None 
        self.clamp_min = clamp_min
        self.clamp_min = -50

        num_states = 2
        alpha_list = torch.nn.init.uniform_(torch.empty(num_states), a=0, b=1) 
        self.alpha_list = nn.Parameter(alpha_list, requires_grad=True)


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
	    num_items_in_batch: int = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        last_state=None
        if past_key_value is not None and len(past_key_value) > self.layer_idx:
            last_state=past_key_value[self.layer_idx]

        gk = rearrange(self.gk_proj(hidden_states), 'b s (h d) -> b h s d', h=self.num_heads)
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        gk2 = rearrange(self.gk_proj2(hidden_states), 'b s (h d) -> b h s d', h=self.num_heads)
        gk2 = F.logsigmoid(gk2) / (self.gate_logit_normalizer)

        if self.clamp_min is not None:
            gk2 = torch.clamp_min(gk2, self.clamp_min)

        recurrent_state = last_state[0] if last_state is not None else None 
        recurrent_state2 = last_state[1] if last_state is not None else None 

        lr_out, recurrent_state_far = fused_recurrent_gla(query_states, key_states, value_states, gk, initial_state=recurrent_state,  output_final_state=True)
        lr_out_2, recurrent_state_close = fused_recurrent_gla(query_states, key_states, value_states, gk2, initial_state=recurrent_state2,  output_final_state=True)

        if past_key_value is not None:
            past_key_value.update(
                recurrent_state_far,
                recurrent_state_close,
                self.layer_idx
            )

        alpha = F.softmax(self.alpha_list, dim=0)
        lr_out_list = [lr_out, lr_out_2]
        attn_output = 0
        for i in range(len(alpha)):
            attn_output += alpha[i] * lr_out_list[i]

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        else:
            attn_weights = (gk.exp(), gk2.exp())

        return attn_output, attn_weights, past_key_value

