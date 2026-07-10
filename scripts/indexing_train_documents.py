from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, NamedTuple, Tuple, Union

from bm25s import BM25
from datasets import load_from_disk
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, Qwen2Tokenizer

# fmt: off
argparser = ArgumentParser("Performs indexing to find sentences similar to the input.")
argparser.add_argument("--input-file", type=str, default="data/preprocessed_data/hanja_train_dataset")
argparser.add_argument("--index-document", action="store_true", help="Store documents into index")
argparser.add_argument("--index-name", type=str, default="train_retriever_0525_10_3")
argparser.add_argument("--tokenizer-cache", type=str, default="tokenizer_for_BM25s_freq_10_qwen_3")
argparser.add_argument("--min-filtering-token-length", type=int, default=5)
argparser.add_argument("--max-filtering-token-length", type=int, default=128)
# fmt: on


class Tokenized(NamedTuple):
    ids: List[List[int]]
    vocab: dict


def filter_sentences(
    data: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerFast, args: Namespace
) -> List[Tuple[str, str]]:
    total_sentences = []

    for datum in tqdm(data, desc="Filtering sentences", mininterval=1):
        hanja_tokens = tokenizer.tokenize(datum["hanja"])

        if args.min_filtering_token_length <= len(hanja_tokens) <= args.max_filtering_token_length:
            total_sentences.append((datum["hanja"], datum["data_id"]))

    return total_sentences


def tokenize_for_BM25s(
    texts: Union[str, List[str]], tokenizer: PreTrainedTokenizerFast
) -> Union[List[List[str]], Tokenized]:

    if isinstance(texts, str):
        texts = [texts]

    token_lists = []

    for text in tqdm(texts, desc="Split strings", disable=True):
        token_lists.append(tokenizer.tokenize(text))

    return token_lists


def index_documents(corpus: List[Tuple[str, str]], model_path: str, tokenizer: PreTrainedTokenizerFast):
    hanja_list = []
    data_ids = []
    for hanja, data_id in corpus:
        hanja_list.append(hanja)
        data_ids.append(data_id)

    document_tokens = tokenize_for_BM25s(hanja_list, tokenizer)

    retriever = BM25(backend="numba")
    retriever.index(document_tokens)

    corpus_dict_list = [{"id": data_id, "text": hanja} for hanja, data_id in zip(hanja_list, data_ids)]
    retriever.save(model_path, corpus=corpus_dict_list)


if __name__ == "__main__":
    args = argparser.parse_args()
    tokenizer = Qwen2Tokenizer.from_pretrained(args.tokenizer_cache, require_fast=True)

    if args.index_document:
        data = load_from_disk(args.input_file)
        sentences = filter_sentences(data, tokenizer, args)
        index_documents(sentences, args.index_name, tokenizer)
