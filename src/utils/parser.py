# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import HfArgumentParser, TrainingArguments

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingArguments(TrainingArguments):

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    tokenizer: str = field(
        default="mistralai/Mistral-7B-v0.1",
        metadata={"help": "Name of the tokenizer to use."}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    from_config: bool = field(
        default=True,
        metadata={"help": "Whether to initialize models from scratch."},
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The dataset(s) to use. Use commas to separate multiple datasets."},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of provided dataset(s) to use."},
    )
    cache_dir: str = field(
        default=None,
        metadata={"help": "Path to the cached tokenized dataset."},
    )
    split: str = field(
        default="train",
        metadata={"help": "Which dataset split to use for training and evaluation."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    buffer_size: int = field(
        default=2048,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."},
    )
    context_length: int = field(
        default=2048,
        metadata={"help": "The context length of the tokenized inputs in the dataset."},
    )
    freeze: bool = field(
        default=False,
        metadata={"help": "Whether to freeze all parameters other than attention"},
    )
    soft_loss: float = field(
        default=0.001,
        metadata={"help": "Final logits alignment. Perhaps somewhere (0.001, 0.01)"},
    )
    soft_loss_decay: float = field(
        default=1.0,
        metadata={"help": "Final logits decay. Default 1.0 (not decaying)"},
    )
    align_loss: float = field(
        default=0.01,
        metadata={"help": "For logits alignment for each layer. Perhaps somewhere (0.01, 0.1)"},
    )
    gla_list: str = field(
        default="",
        metadata={"help": "Specify layers to convert"},
    )
    loss_func: str = field(
        default="mse",
        metadata={"help": "Specify layers to convert"},
    )
    min_lr_rate: float = field(
        default=0.1,
        metadata={"help": "For logits alignment for each layer. Perhaps somewhere (0.01, 0.1)"},
    )
    use_cont_loss: bool = field(
        default=False,
        metadata={"help": "Use contrastive loss instead as soft loss"},
    )






def get_train_args():
    parser = HfArgumentParser(TrainingArguments)
    args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if unknown_args:
        print(parser.format_help())
        print("Got unknown args, potentially deprecated arguments: {}".format(unknown_args))
        raise ValueError("Some specified arguments are not used by the HfArgumentParser: {}".format(unknown_args))

    if args.should_log:
        transformers.utils.logging.set_verbosity(args.get_process_log_level())
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    # set seeds manually
    transformers.set_seed(args.seed)
    return args
