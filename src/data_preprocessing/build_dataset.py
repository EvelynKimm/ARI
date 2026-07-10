import gc
import random
import time
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import ujson as json
from bm25s import BM25
from datasets import Dataset, concatenate_datasets, load_from_disk
from loguru import logger
from transformers import PreTrainedTokenizerFast, Qwen2Tokenizer

from src.data_preprocessing.mask import MaskConverter
from src.hanja_llm.retrieve_documents import (
    retrieve_random_sentences,
    retrieve_related_sentences,
    retrieve_related_sentences_wo_fuzz,
)

# fmt: off
argparser = ArgumentParser("Builds a dataset by finding relevant sentences for each data entry.")
argparser.add_argument("--train-data-path", type=str, default="data/preprocessed_data/hanja_train_dataset.json")
argparser.add_argument("--valid-data-path", type=str, default="data/preprocessed_data/hanja_valid_dataset.json")
argparser.add_argument("--test-data-path", type=str, default="data/preprocessed_data/hanja_test_dataset.json")
argparser.add_argument("--output-data-path", type=str, default="data/preprocessed_data/dataset/")
argparser.add_argument("--train-output-file", type=str, default="train_dataset")
argparser.add_argument("--valid-output-file", type=str, default="valid_dataset")
argparser.add_argument("--test-output-file", type=str, default="test_dataset")

argparser.add_argument("--max-train-data", type=int, default=100)
argparser.add_argument("--max-valid-data", type=int, default=100)
argparser.add_argument("--max-test-data", type=int, default=100)

argparser.add_argument("--num-candidates", type=int, default=32)
argparser.add_argument("--num-selected", type=int, default=20)

argparser.add_argument("--index-name", type=str, default="train_retriever_0525_10_2p5")
argparser.add_argument("--tokenizer-cache", type=str, default="tokenizer/tokenizer_for_BM25s_freq_10_qwen_2p5")

argparser.add_argument("--random-seed", type=int, default=42)
argparser.add_argument("--fuzz-threshold", type=int, default=80)
argparser.add_argument("--eval-average-mask-ratio", type=float, default=0.03)
argparser.add_argument("--max-mask-ratio", type=float, default=0.15)

argparser.add_argument("--batch-size", type=int, default=32)
argparser.add_argument("--num-workers", type=int, default=20)

argparser.add_argument("--use-ner-vocab", action="store_true")
argparser.add_argument("--ner-vocab-percentage", type=float, default=1.0)

argparser.add_argument("--do-train", action="store_true", help="train 데이터셋만 생성")
argparser.add_argument("--do-valid", action="store_true", help="valid 데이터셋만 생성")
argparser.add_argument("--do-test", action="store_true", help="test 데이터셋만 생성")

argparser.add_argument("--masking-strategy", type=str, default="mixed")
# fmt: on


def build_ner_vocab(data: List[Dict[str, Any]]) -> List[str]:
    ner_set = set()
    for item in data:
        for tag in item["ner"]:
            ner_set.add(tag)
    return list(ner_set)


def format_documents(docs: List[str]) -> str:
    return "\n".join(f"- {d}" for d in docs)


def save_hf_part_100k(examples, base_dir, dataset_name, part_idx):
    root = Path(base_dir) / dataset_name
    part_dir = root / f"{dataset_name}_part_{part_idx}"
    part_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(examples).save_to_disk(str(part_dir))
    return str(part_dir)


def merge_hf_parts(base_dir, dataset_name, final_subdir="_merged"):
    root = Path(base_dir) / dataset_name
    parts = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith(f"{dataset_name}_part_")],
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not parts:
        raise FileNotFoundError(f"No parts under {root}")
    merged = load_from_disk(str(parts[0]))
    for p in parts[1:]:
        merged = concatenate_datasets([merged, load_from_disk(str(p))])
    out_dir = root / final_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_to_disk(str(out_dir))
    return str(out_dir)


def run_pipeline_in_fixed_100k_parts(data, *, output_dataset_name, output_base_dir, build_fn):
    BATCH, n = 100_000, len(data)
    for i in range((n + BATCH - 1) // BATCH):
        s, e = i * BATCH, min((i + 1) * BATCH, n)
        built = build_fn(data[s:e])
        if built:
            save_hf_part_100k(built, output_base_dir, output_dataset_name, i)
    return merge_hf_parts(output_base_dir, output_dataset_name)


def process_batch(
    batch: List[Dict[str, Any]],
    retriever: BM25,
    tokenizer: PreTrainedTokenizerFast,
    mask_converter: MaskConverter,
    args: Namespace,
    preloaded_corpus: List[Any],
) -> List[Dict[str, Any]]:

    local_rng = random.Random(args.random_seed)

    new_batch = []
    for item in batch:
        new_item = dict(item)
        for k in ("korean", "english"):
            if k in new_item:
                del new_item[k]

        sample_ner = new_item.get("ner", [])
        masked_document, mask_label = mask_converter.mask_document(new_item["hanja"], sample_ner)

        new_item["masked_document"] = masked_document
        new_item["mask_label"] = json.dumps(mask_label, ensure_ascii=False)

        data_id = new_item["data_id"]
        input_doc = new_item["hanja"]
        related_docs = retrieve_related_sentences(
            input_doc,
            masked_document,
            retriever,
            tokenizer,
            threshold=args.fuzz_threshold,
            k=args.num_candidates,
            query_data_id=data_id,
        )
        new_item["related_documents_fuzz80"] = related_docs
        new_item["related_documents_fuzz80_top20"] = format_documents(related_docs[: args.num_selected])

        related_docs_wo_fuzz = retrieve_related_sentences_wo_fuzz(
            input_doc, masked_document, retriever, tokenizer, k=args.num_candidates, query_data_id=data_id
        )
        new_item["related_documents_wo_fuzz"] = format_documents(related_docs_wo_fuzz)
        new_item["related_documents_wo_fuzz_top20"] = format_documents(related_docs_wo_fuzz[: args.num_selected])

        random_docs = retrieve_random_sentences(preloaded_corpus, local_rng, args.num_selected)
        new_item["random_documents"] = format_documents(random_docs)

        new_batch.append(new_item)

    return new_batch


def run_pipeline(
    data,
    *,
    retriever,
    tokenizer,
    ner_vocab,
    args,
):
    use_ner_vocab = bool(getattr(args, "use_ner_vocab", False) and ner_vocab)

    conv_vocab = MaskConverter(
        average_mask_ratio=args.eval_average_mask_ratio,
        max_mask_ratio=args.max_mask_ratio,
        ner_vocab=ner_vocab,
        seed=args.random_seed,
        mode="dn",
        masking_strategy=args.masking_strategy,
    )
    conv_none = MaskConverter(
        average_mask_ratio=args.eval_average_mask_ratio,
        max_mask_ratio=args.max_mask_ratio,
        ner_vocab=None,
        seed=args.random_seed,
        mode="dn",
    )

    preloaded_corpus = list(retriever.corpus)
    batches = [data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]

    def process_one_batch(batch_records):
        converter = conv_vocab if use_ner_vocab else conv_none
        return process_batch(
            batch=batch_records,
            retriever=retriever,
            tokenizer=tokenizer,
            mask_converter=converter,
            args=args,
            preloaded_corpus=preloaded_corpus,
        )

    final = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        for res in pool.map(process_one_batch, batches):
            final.extend(res)
    return final


def process_split(data: List[Dict[str, Any]], max_n: int, args: Namespace, output_prefix: str):
    ner_tags = build_ner_vocab(data)
    n_total = len(data)
    n_vocab = min(int(max_n * args.ner_vocab_percentage), n_total)
    n_none = max(0, max_n - n_vocab)
    logger.info(
        f"[process_split] total={n_total} max_n={max_n} "
        f"ner_vocab_percentage={args.ner_vocab_percentage} "
        f"n_vocab={n_vocab} n_none={n_none}"
    )

    t0 = time.time()

    if n_vocab > 0:
        args_vocab = deepcopy(args)
        args_vocab.use_ner_vocab = True

        def build_vocab_fn(chunk):
            return run_pipeline(
                data=chunk, retriever=retriever, tokenizer=tokenizer, ner_vocab=ner_tags, args=args_vocab
            )

        run_pipeline_in_fixed_100k_parts(
            data=data[:n_vocab],
            output_dataset_name=f"{output_prefix}_vocab",
            output_base_dir=args.output_data_path,
            build_fn=build_vocab_fn,
        )
        ds_vocab = load_from_disk(str(Path(args.output_data_path) / f"{output_prefix}_vocab" / "_merged"))
    else:
        ds_vocab = None

    if n_none > 0:
        args_none = deepcopy(args)
        args_none.use_ner_vocab = False

        def build_none_fn(chunk):
            return run_pipeline(data=chunk, retriever=retriever, tokenizer=tokenizer, ner_vocab=None, args=args_none)

        run_pipeline_in_fixed_100k_parts(
            data=data[n_vocab:max_n],
            output_dataset_name=f"{output_prefix}_none",
            output_base_dir=args.output_data_path,
            build_fn=build_none_fn,
        )
        ds_none = load_from_disk(str(Path(args.output_data_path) / f"{output_prefix}_none" / "_merged"))
    else:
        ds_none = None

    if ds_vocab is not None and ds_none is not None:
        final = concatenate_datasets([ds_vocab, ds_none])
    elif ds_vocab is not None:
        final = ds_vocab
    elif ds_none is not None:
        final = ds_none
    else:
        raise ValueError("[WARN] 생성된 데이터가 없습니다 (ds_vocab, ds_none 모두 None)")

    final_save_path = Path(args.output_data_path) / output_prefix / "_merged"
    final_save_path.parent.mkdir(parents=True, exist_ok=True)
    final_save_path.mkdir(exist_ok=True)
    final.save_to_disk(str(final_save_path))

    t1 = time.time()
    logger.info(f"⏰ 데이터셋 구축 소요시간 : {t1 - t0:.2f}초")
    logger.info(
        f"📊 최종: {len(final)}건 | "
        f"NER vocab: {len(ds_vocab) if ds_vocab else 0}건 | "
        f"NER none: {len(ds_none) if ds_none else 0}건 | "
        f"저장: {final_save_path}"
    )

    del ds_vocab, ds_none, final
    gc.collect()


if __name__ == "__main__":
    args = argparser.parse_args()
    random.seed(args.random_seed)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.tokenizer_cache, require_fast=True)
    retriever = BM25(dtype="float16", backend="numba").load(args.index_name, mmap=True, load_corpus=True)

    if args.do_train:
        with open(args.train_data_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        process_split(train_data, args.max_train_data, args, args.train_output_file)

    if args.do_valid:
        with open(args.valid_data_path, "r", encoding="utf-8") as f:
            valid_data = json.load(f)
        process_split(valid_data, args.max_valid_data, args, args.valid_output_file)

    if args.do_test:
        with open(args.test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        process_split(test_data, args.max_test_data, args, args.test_output_file)
