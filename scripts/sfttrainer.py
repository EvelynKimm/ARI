import datetime
import math
import os
from argparse import ArgumentParser
from pathlib import Path

import psutil
import torch
import torch.distributed as dist
from datasets import load_from_disk
from loguru import logger
from peft import LoraConfig
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

import wandb
from src.hanja_llm.dataset_util import CollateFn, LengthBasedBatchSampler
from src.hanja_llm.hanja_restoration_dataset import LABEL_MASK_ID, HanjaRestorationDatasetForTrain

# fmt: off
argparser = ArgumentParser("Train script for Hanja restoration with Hugging Face SFTTrainer")
argparser.add_argument("--model-uri", type=str, default="Qwen/Qwen3-32B", help="Model name or path.")
argparser.add_argument("--train-data-ep0", type=str, required=True, help="Epoch 0 train 데이터 경로")
argparser.add_argument("--train-data-ep1", type=str, required=True, help="Epoch 1 train 데이터 경로")
argparser.add_argument("--valid-data", type=str, default="data/preprocessed_data/dataset/non_reasoning_dataset/valid_ner_vocab_100")
argparser.add_argument("--max-length", type=int, default=4096)
argparser.add_argument("--add-date-info", action="store_true")

argparser.add_argument("--train-batch-size", type=int, default=2)
argparser.add_argument("--valid-batch-size", type=int, default=6)
argparser.add_argument("--gradient-accumulation-steps", type=int, default=1)
argparser.add_argument("--optim", type=str, default="lion_32bit")
argparser.add_argument("--epochs", type=int, default=1)
argparser.add_argument("--max-lr", type=float, default=1e-5)
argparser.add_argument("--lr-warmup-ratio", type=float, default=0.05)
argparser.add_argument("--weight-decay", type=float, default=1e-2)

argparser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
argparser.add_argument("--log-dir", type=str, default="logs")
argparser.add_argument("--run-name", type=str, required=True)
argparser.add_argument("--logging-interval", type=int, default=5)
argparser.add_argument("--eval-interval", type=int, default=15)
argparser.add_argument("--model-save-interval", type=int, default=1000)
argparser.add_argument("--max-steps", type=int, default=100000)
argparser.add_argument("--seed", type=int, default=42)
argparser.add_argument("--use-hanja-tokenizer", action="store_true", help="Load and use the Hanja tokenizer")
argparser.add_argument("--resume-from-checkpoint", action="store_true", help="If set, resume training from the latest checkpoint in output_dir")
argparser.add_argument("--cpu-offload", action="store_true", help="If set, cpu_offload = True")
argparser.add_argument("--lora-rank", type=int, default=16)
argparser.add_argument("--lora-alpha", type=int, default=32)
argparser.add_argument("--use-lora", action="store_true", help="Enable LoRA adaptation")
argparser.add_argument("--use-rslora", action="store_true", help="Enable rsLoRA adaptation")
# fmt: on


def init_distributed():
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup():
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        local_rank = init_distributed()
        torch.distributed.barrier(device_ids=[local_rank])


def cleanup():
    dist.destroy_process_group()


def setup_omp_threads_for_fsdp():
    world_size = dist.get_world_size()

    phys_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)

    threads_per_proc = max(1, phys_cores // world_size)
    os.environ["OMP_NUM_THREADS"] = str(threads_per_proc)

    return threads_per_proc


def load_dataset(args, dataset_type, tokenizer):
    if dataset_type == "train":
        data_path = args.train_data
    elif dataset_type == "valid":
        data_path = args.valid_data
    else:
        raise ValueError("Dataset Name Error")

    data = load_from_disk(data_path)

    dataset = HanjaRestorationDatasetForTrain(data, tokenizer, args.max_length)

    return dataset


class PerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            ppl = math.exp(logs["loss"])
            logs["perplexity"] = ppl
            wandb.log({"perplexity": ppl})
        if "eval_loss" in logs:
            eval_ppl = math.exp(logs["eval_loss"])
            logs["eval_perplexity"] = eval_ppl
            wandb.log({"eval_perplexity": eval_ppl})

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or "eval_loss" not in metrics:
            return
        metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])


class EpochTrainDatasetCallback(TrainerCallback):
    def __init__(self, trainer, args, train_list):
        super().__init__()
        self.trainer = trainer
        self.args = args
        self.train_list = list(train_list)
        self.applied_epoch = None

    def _swap_dataset_for(self, epoch_idx: int):
        idx = min(epoch_idx, len(self.train_list) - 1)
        self.args.train_data = self.train_list[idx]

        pre_dataset = load_from_disk(self.args.train_data)
        dataset = HanjaRestorationDatasetForTrain(pre_dataset, self.trainer.tokenizer, self.args.max_length)

        self.trainer.train_dataset = dataset
        self.trainer._train_dataloader = None
        if self.trainer.is_world_process_zero():
            logger.info(f"[EpochSwitch] epoch {epoch_idx} -> {self.args.train_data} (len={len(dataset)})")

    def on_train_begin(self, args, state, control, **kwargs):
        self._swap_dataset_for(0)
        self.applied_epoch = 0
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        curr = int(state.epoch or 0)
        if self.applied_epoch != curr:
            self._swap_dataset_for(curr)
            self.applied_epoch = curr
        return control


class MySFTTrainer(SFTTrainer):
    def __init__(self, *args, tokenizer, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def _prepare_dataset(self, dataset, *args, **kwargs):
        if not hasattr(dataset, "map"):
            return dataset
        return super()._prepare_dataset(dataset, *args, **kwargs)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        sampler = LengthBasedBatchSampler(
            [d[0].size(0) for d in self.train_dataset],
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            batch_size_per_device=self.args.per_device_train_batch_size,
            shuffle=True,
            seed=self.args.seed,
            drop_last=True,
        )

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.data_collator,
            persistent_workers=self.args.dataloader_persistent_workers,
        )

        logger.info(f"Number of [DataLoader workers: {self.args.dataloader_num_workers}")
        logger.info(f"Size of train dataLoader: {len(train_dataloader)}")

        return train_dataloader

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self._eval_dataloaders[dataloader_key]

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        sampler = LengthBasedBatchSampler(
            [d[0].size(0) for d in eval_dataset],
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            batch_size_per_device=self.args.per_device_eval_batch_size,
            shuffle=False,
            seed=self.args.seed,
            drop_last=True,
        )

        valid_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            sampler=sampler,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.data_collator,
            persistent_workers=self.args.dataloader_persistent_workers,
        )

        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = valid_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: valid_dataloader}

        logger.info(f"Size of valid dataLoader: {len(valid_dataloader)}")
        return valid_dataloader

    def log_metrics(self, split: str, metrics: dict) -> None:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        cpu_rss_mb = mem_info.rss / (1024**2)

        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            stats = torch.cuda.memory_stats(dev)
            gpu_alloc_mb = torch.cuda.memory_allocated(dev) / (1024**2)
            gpu_reserved_mb = torch.cuda.memory_reserved(dev) / (1024**2)
            gpu_peak_mb = torch.cuda.max_memory_allocated(dev) / (1024**2)
            gpu_active_peak_mb = stats.get("active_bytes.all.peak", 0) / (1024**2)
            gpu_reserved_peak_mb = stats.get("reserved_bytes.all.peak", 0) / (1024**2)
            gpu_inactive_split_mb = stats.get("inactive_split_bytes.all.current", 0) / (1024**2)
            gpu_num_ooms = int(stats.get("num_ooms", 0))
        else:
            gpu_alloc_mb = gpu_reserved_mb = gpu_peak_mb = 0.0
            gpu_active_peak_mb = gpu_reserved_peak_mb = gpu_inactive_split_mb = 0.0
            gpu_num_ooms = 0

        metrics.update(
            {
                "mem_cpu_rss_mb": cpu_rss_mb,
                "mem_gpu_alloc_mb": gpu_alloc_mb,
                "mem_gpu_reserved_mb": gpu_reserved_mb,
                "mem_gpu_peak_mb": gpu_peak_mb,
                "mem_gpu_active_peak_mb": gpu_active_peak_mb,
                "mem_gpu_reserved_peak_mb": gpu_reserved_peak_mb,
                "mem_gpu_inactive_split_mb": gpu_inactive_split_mb,
                "mem_gpu_num_ooms": gpu_num_ooms,
            }
        )

        if self.is_world_process_zero():
            print(f"[{split} Metrics] {metrics}")


def main():
    args = argparser.parse_args()

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
    args.log_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    args.train_data = args.train_data_ep0

    logger.add(os.path.join(args.log_dir, "train.log"), level="INFO")

    setup()
    set_seed(args.seed)

    threads = setup_omp_threads_for_fsdp()
    logger.info(f"OMP_NUM_THREADS : {threads}")

    if args.use_hanja_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("./tokenizer_for_NER", trust_remote_code=True, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_uri, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_uri,
        low_cpu_mem_usage=True,
        attn_implementation="kernels-community/vllm-flash-attn3",
        torch_dtype=torch.bfloat16,
    )

    model.config.use_cache = False
    logger.info(f"Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.use_hanja_tokenizer:
        original_vocab_size = model.config.vocab_size
        new_vocab_size = len(tokenizer)
        # 8291개 Hanja token 추가
        logger.info(f"Resizing token embeddings from {original_vocab_size} to {new_vocab_size}")
        model.resize_token_embeddings(new_vocab_size, mean_resizing=True)

    train_dataset = load_dataset(args, "train", tokenizer)
    valid_dataset = load_dataset(args, "valid", tokenizer)

    if int(os.environ.get("RANK", "0")) == 0:
        wandb.init(
            project="joseon-analysis-v2",
            name=args.run_name,
        )
    else:
        wandb.init(mode="disabled")

    train_list = [args.train_data_ep0, args.train_data_ep1]

    training_args = SFTConfig(
        output_dir=args.checkpoint_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.valid_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=1,
        eval_strategy="steps",
        eval_steps=args.eval_interval,
        logging_strategy="steps",
        logging_steps=args.logging_interval,
        save_strategy="steps",
        save_steps=args.model_save_interval,
        report_to="wandb",
        run_name=args.run_name,
        seed=args.seed,
        max_grad_norm=0.5,
        dataloader_persistent_workers=True,
        dataloader_num_workers=2,
        gradient_checkpointing=False,
        optim=args.optim,
        learning_rate=args.max_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.lr_warmup_ratio,
        weight_decay=args.weight_decay,
        average_tokens_across_devices=False,
        prediction_loss_only=True,
        save_only_model=True,
        label_names=["labels"],
        disable_tqdm=False,
        remove_unused_columns=False,
        bf16=False,
        fp16=False,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "activation_checkpointing": True,
            "auto_wrap_policy": "transformer_based_wrap",
            "cpu_offload": args.cpu_offload,
            "fsdp_cpu_ram_efficient_loading": True,
        },
    )

    data_collator = CollateFn(
        tokenizer.encode(tokenizer.special_tokens_map["pad_token"])[0],
        LABEL_MASK_ID,
        args.max_length,
    )

    peft_config = None
    if args.use_lora or args.use_rslora:
        peft_config = LoraConfig(
            use_rslora=args.use_rslora,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            r=args.lora_rank,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        mode = "rsLoRA" if args.use_rslora else "LoRA"
        logger.info(f"{mode} adaptation enabled (lora rank: {args.lora_rank}, lora alpha: {args.lora_alpha})")

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "tokenizer": tokenizer,
        "data_collator": data_collator,
        "callbacks": [PerplexityCallback()],
    }

    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = MySFTTrainer(**trainer_kwargs)

    if not trainer.is_world_process_zero():
        logger.disable("")

    trainer.add_callback(EpochTrainDatasetCallback(trainer, args, train_list))

    logger.info(f"Size of train dataset: {len(train_dataset)}")
    logger.info(f"Size of valid dataset: {len(valid_dataset)}")

    torch.cuda.empty_cache()

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger.warning(f"FSDP state_dict_type = {getattr(trainer.accelerator.state.fsdp_plugin, 'state_dict_type', 'N/A')}")

    trainer.save_model()
    trainer.save_state()

    if trainer.is_world_process_zero():
        ckpt_dir = os.path.join(args.checkpoint_dir, f"checkpoint-{trainer.state.global_step}")
        tokenizer.save_pretrained(ckpt_dir)

        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        trainer.accelerator.unwrap_model(trainer.model).config.to_json_file(os.path.join(ckpt_dir, "config.json"))

    trainer.log_metrics("train", train_result.metrics)

    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    try:
        metrics["perplexity"] = math.exp(metrics["eval_loss"])
    except OverflowError:
        metrics["perplexity"] = float("inf")

    trainer.log_metrics("eval", metrics)

    logger.info("Done!")

    cleanup()


if __name__ == "__main__":
    main()
