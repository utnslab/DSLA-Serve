# -*- coding: utf-8 -*-

from datasets import load_from_disk
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          Trainer)
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb, LlamaDecoderLayer
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig
from typing import Optional, Tuple, Union 

import torch
import torch.nn as nn

from utils.data import DataCollatorForLanguageModeling
from utils.logging import LogCallback, get_logger
from utils.parser import get_train_args

from einops import rearrange
import torch.nn.functional as F
from small_fla import fused_recurrent_gla
from distill import DistillTrainer

from safetensors import safe_open
from pathlib import Path

from layer import convert_transformer_to_dsla_multiple, DualStateLinearAttention

logger = get_logger(__name__)

def main():
    args = get_train_args()
    logger.info(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=True,
        add_bos_token=True,
        add_eos_token=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    teacher = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.config.output_hidden_states = True
    model.config.output_attentions = True
    teacher.config.output_hidden_states = True
    config = model.config
    config.kernel_hidden_size=teacher.config.hidden_size//teacher.config.num_attention_heads

    config.dsla_list = args.dsla_list 
    config.layer_to_train = -1 

    print("Converting Baseline llama to DSLA...")
    device = model.device
    print("Model's device:", device)
    model = convert_transformer_to_dsla_multiple(model, config)
    model = model.to(torch.bfloat16)
    model = model.to('cuda')
    teacher = teacher.to(torch.bfloat16)
    teacher = teacher.to('cuda')
    model.train()
    teacher.eval()

    if args.freeze:
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
            if not 'self_attn' in name and not 'attention' in name:
                param.requires_grad = False
            if 'self_attn' in name or 'attention' in name:
                layer_id = int(name.split('.')[2])
                if layer_id not in config.dsla_list:
                    param.requires_grad = False

    trainable_params, all_param = model.num_parameters(only_trainable=True), model.num_parameters()
    dataset = load_from_disk(args.cache_dir)
    dataset = dataset.shuffle(seed=args.seed)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    if args.lr_scheduler_type == 'cosine_with_min_lr':
        args.lr_scheduler_kwargs = {'min_lr_rate': args.min_lr_rate}
    if args.lr_scheduler_type == 'warmup_stable_decay':
        args.lr_scheduler_kwargs = {
            'num_stable_steps': args.max_steps * 0.9 - args.warmup_steps,
            'num_decay_steps': args.max_steps * 0.1
        }

    trainer = DistillTrainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LogCallback()],
        train_dataset=dataset
    )

    final_logits_lambda = args.soft_loss 
    inner_logits_lambda = args.align_loss
    alpha_decay = args.soft_loss_decay
    temperature = 1.0
    trainer.register_teacher_model(teacher, final_logits_lambda, inner_logits_lambda, temperature=temperature, alpha_decay=alpha_decay,
                                    loss_func=args.loss_func, layer_to_train=config.layer_to_train, cont_loss=args.use_cont_loss, dsla_list=config.dsla_list)

    results = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(trainer.args.output_dir)

    trainer.log_metrics("train", results.metrics)
    trainer.save_metrics("train", results.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
