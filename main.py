from alpaca_dataset import AlpacaData, split_train_val
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from alpaca_trainer import AlpacaTrainer
from datetime import datetime
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import torch
import sys
import transformers
import logging


if __name__ == "__main__":
    # swap to alpaca
    model_name = "gpt2"
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I_%M_%S_%p")
    train_args = {
        "seed": 42,
        "log_level": "INFO",
        "fp16": True,
        "bf16": False,
        "fsdp": ["full_shard", "offload"],
        "optim": "adamw_torch",
        "lr_scheduler_type": "linear",
        "weight_decay": 1e-4,
        "learning_rate": 5e-5,
        "num_train_epochs": 6,
        "local_rank": int(os.environ["LOCAL_RANK"]),
        "output_dir": f"./outputs/{model_name}/{date_of_run}",
        "overwrite_output_dir": True,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "logging_strategy": "steps",
        "disable_tqdm": False,
        "logging_dir": "logs",
        "logging_steps": 20,
        "evaluation_strategy": "epoch",
        "gradient_accumulation_steps": 2,
        "do_train": True,
    }

    set_seed(train_args["seed"])
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    transformers.utils.logging.set_verbosity_info()
    log_level = train_args["log_level"]
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    json_data = load_dataset("json", data_dir="data")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if train_args["fp16"] else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    block_size = tokenizer.model_max_length
    train, val, test = split_train_val(json_data["train"], tokenizer)
    # alpaca = AlpacaTrainer(model, block_size, tokenizer, train, val)
    # alpaca.train(train_args)
