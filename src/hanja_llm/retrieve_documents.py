import random
import re
from typing import Any, List

from bm25s import BM25

from rapidfuzz import fuzz
from transformers import PreTrainedTokenizerFast

from scripts.indexing_train_documents import tokenize_for_BM25s

DAMAGE_PATTERNS = re.compile(r"\[D\d+\](?:\[D\d+\])*")
EXCLUDE_CHARS = {".", ",", ":", ";", " ", "·", "?", "'", '"', "(", ")", "{", "}", "<", ">"}


def remove_damage_patterns(text: str) -> str:
    return re.sub(DAMAGE_PATTERNS, " ", text)


def are_sentences_equal(s1: str, s2: str) -> bool:
    def clean(text: str) -> str:
        return "".join(ch for ch in text if ch not in EXCLUDE_CHARS)

    return clean(s1) == clean(s2)


def retrieve_related_sentences(
    input_document: str,
    query: str,
    retriever: BM25,
    tokenizer: PreTrainedTokenizerFast,
    threshold: int,
    k: int = 8,
    query_data_id: str = None,
) -> List[str]:
    cleaned_query = remove_damage_patterns(query)
    query_tokens = tokenize_for_BM25s(cleaned_query, tokenizer)

    top_results = []
    added_texts = set()

    retriever.activate_numba_scorer()
    num_to_retrieve = k * 128

    results, scores = retriever.retrieve(
        query_tokens,
        k=num_to_retrieve,
        corpus=retriever.corpus,
        backend_selection="numba",
        n_threads=16,
    )

    for doc, score in zip(results[0], scores[0]):
        doc_id = doc["id"]
        doc_text = doc["text"]

        if query_data_id is not None and doc_id == query_data_id:
            continue

        if are_sentences_equal(doc_text, input_document):
            continue

        if all(fuzz.ratio(doc_text, prev_text) <= threshold for prev_text in added_texts):
            top_results.append(doc_text)
            added_texts.add(doc_text)

        if len(top_results) >= k:
            break

    return top_results


def retrieve_related_sentences_wo_fuzz(
    input_document: str,
    query: str,
    retriever: BM25,
    tokenizer: PreTrainedTokenizerFast,
    k: int = 8,
    query_data_id: str = None,
) -> List[str]:
    cleaned_query = remove_damage_patterns(query)
    query_tokens = tokenize_for_BM25s(cleaned_query, tokenizer)

    top_results = []
    seen_texts = set()

    retriever.activate_numba_scorer()

    num_to_retrieve = k * 128

    results, scores = retriever.retrieve(
        query_tokens,
        k=num_to_retrieve,
        corpus=retriever.corpus,
        backend_selection="numba",
        n_threads=16,
    )

    for doc, score in zip(results[0], scores[0]):
        doc_id = doc["id"]
        doc_text = doc["text"]

        if query_data_id is not None and doc_id == query_data_id:
            continue

        if are_sentences_equal(doc_text, input_document):
            continue

        if doc_text in seen_texts:
            continue
        top_results.append(doc_text)
        seen_texts.add(doc_text)

        if len(top_results) >= k:
            break

    final_results = top_results[:k]

    return final_results


def retrieve_random_sentences(corpus: List[Any], rng: random.Random, max_num_related_documents: int = 8) -> List[str]:
    selected_docs = rng.sample(corpus, max_num_related_documents)
    random_texts = [doc["text"] for doc in selected_docs]

    return random_texts
