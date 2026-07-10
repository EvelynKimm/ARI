import os
import random
from argparse import ArgumentParser, Namespace
from collections import Counter, defaultdict
from functools import partial
from glob import glob
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple

import tiktoken
import ujson as json
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data_preprocessing.utils import DAMAGE_TOKEN, KING_KOREAN_TO_HANJA, process_ner_texts, process_text

argparser = ArgumentParser("Preprocess Hanja documents.")
argparser.add_argument("--jrs-file-pattern", type=str, default="data/jrs/*/*/*.json")
argparser.add_argument("--ajd-file-pattern", type=str, default="data/ajd/*/*/*.json")
argparser.add_argument("--output-dir", type=str, default="data/preprocessed_data")
argparser.add_argument("--tokenizer", type=str, default="o200k_base")
argparser.add_argument("--min-ner-length", type=int, default=2)
argparser.add_argument("--max-ner-length", type=int, default=5)
argparser.add_argument("--min-ner-vocab-freq", type=int, default=20)
argparser.add_argument("--min-length", type=int, default=10)
argparser.add_argument("--max-num-hanja-tokens", type=int, default=1024)
argparser.add_argument("--translation-validation-dataset-size", type=int, default=1000)
argparser.add_argument("--translation-test-dataset-size", type=int, default=1000)
argparser.add_argument("--restoration-test-dataset-size", type=int, default=10000)
argparser.add_argument("--restoration-valid-dataset-size", type=int, default=10000)
argparser.add_argument("--num-workers", type=int, default=8)
argparser.add_argument("--seed", type=int, default=42)


def process_one_file(
    file_path: str, min_length: int = 10, min_ner_length: int = 2, max_ner_length: int = 8
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    한 파일에 있는 데이터들을 읽어서 전처리 후 damaged_documents와 undamaged_documents로 나눕니다.
    """
    undamaged_documents = []
    damaged_documents = []

    with open(file_path) as f:
        data = json.load(f)
        data_type = data["data_type"]
        king = data["king"]
        year = data["real_year"]
        month = data["month"]
        for date, day_texts in data["documents"].items():
            for document_idx, document in enumerate(day_texts):
                hanja_text = process_text(
                    document["hanja"]
                )  # 이거를 english text도 해야한다. 이거를 영어도 하니까 process_text로 변수명 바꾸면 good
                if len(hanja_text) < min_length:
                    continue

                korean_text: Optional[str] = document.get("korean")
                if korean_text is not None:
                    # 한글 정제
                    korean_text = process_text(korean_text)
                    if len(korean_text) < min_length:
                        korean_text = None

                english_text: Optional[str] = document.get("english")
                if english_text is not None:
                    # 이 경우에, pre-processing 해주기
                    # 영어 정제
                    english_text = process_text(english_text)
                    # 대소문자 처리 : 영어 text의 첫 글자를 대문자로, 나머지 글자를 소문자로
                    english_text = english_text.capitalize()
                    if len(english_text) < min_length:
                        english_text = None

                year = year.replace("년", "")
                month = month.replace("월", "")
                date = date.replace("일", "")

                data_id = f"{data_type}_{year}-{month}-{date}_{document_idx}"

                output = {
                    "data_type": data_type,
                    "data_id": data_id,
                    "king": KING_KOREAN_TO_HANJA[king],
                    "year": year,
                    "month": month,
                    "date": date,
                    "document_idx": document_idx,
                    "hanja": hanja_text,
                    "korean": korean_text,
                    "english": english_text,
                }

                if DAMAGE_TOKEN in hanja_text:
                    # 한자 문장에 훼손된 글자가 있으면 damaged_documents에
                    damaged_documents.append(output)
                else:
                    ner_word_list = process_ner_texts(document["ner"], min_ner_length, max_ner_length)
                    output["ner"] = [ner_word for ner_word in ner_word_list if DAMAGE_TOKEN not in ner_word]
                    undamaged_documents.append(output)

    return damaged_documents, undamaged_documents


def process_files(args: Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    jrs_files: List[str] = glob(args.jrs_file_pattern)
    ajd_files: List[str] = glob(args.ajd_file_pattern)

    total_files = jrs_files + ajd_files

    undamaged_documents = []
    damaged_documents = []

    with Pool(args.num_workers) as pool:
        process_one_file_fn = partial(
            process_one_file,
            min_length=args.min_length,
            min_ner_length=args.min_ner_length,
            max_ner_length=args.max_ner_length,
        )
        for sub_damaged_documents, sub_undamaged_documents in tqdm(
            pool.imap_unordered(process_one_file_fn, total_files, chunksize=100),
            desc="Process files",
            total=len(total_files),
            mininterval=1,
        ):
            damaged_documents.extend(sub_damaged_documents)
            undamaged_documents.extend(sub_undamaged_documents)

    return undamaged_documents, damaged_documents


if __name__ == "__main__":
    args = argparser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    tokenizer = tiktoken.get_encoding(args.tokenizer)

    undamaged_documents, damaged_documents = process_files(args)

    # 각 왕별 문서 개수 계산
    damaged_document_king_counts = defaultdict(int)
    undamaged_documents_king_counts = defaultdict(int)

    total_num_damaged_characters = 0
    average_damaged_character_ratio = 0
    for document in damaged_documents:
        damaged_document_king_counts[document["king"]] += 1
        num_damaged_characters = document["hanja"].count(DAMAGE_TOKEN)
        total_num_damaged_characters += num_damaged_characters
        average_damaged_character_ratio += num_damaged_characters / len(document["hanja"])

    average_damaged_character_ratio /= len(damaged_documents)

    for document in undamaged_documents:
        undamaged_documents_king_counts[document["king"]] += 1

    logger.info(
        f"Total damaged documents: {sum(damaged_document_king_counts.values())}, "
        f"Total damaged characters: {total_num_damaged_characters}, "
        f"Average damaged character ratio: {average_damaged_character_ratio * 100:.2f}%"
    )
    for king_korean, king_hanja in KING_KOREAN_TO_HANJA.items():
        logger.info(f"{king_korean}: {damaged_document_king_counts[king_hanja]}")

    logger.info(f"Total undamaged documents: {sum(undamaged_documents_king_counts.values())}")
    for king_korean, king_hanja in KING_KOREAN_TO_HANJA.items():
        logger.info(f"{king_korean}: {undamaged_documents_king_counts[king_hanja]}")

    # undamaged_documents에서 영어 데이터 있는 것과 없는 것 분리
    documents_with_english = [
        document
        for document in tqdm(undamaged_documents, desc="Process documents with English", mininterval=1)
        if document["english"] is not None and len(tokenizer.encode(document["hanja"])) <= args.max_num_hanja_tokens
    ]
    documents_without_english = [
        document
        for document in tqdm(undamaged_documents, desc="Process documents without English", mininterval=1)
        if document["english"] is None and len(tokenizer.encode(document["hanja"])) <= args.max_num_hanja_tokens
    ]

    # 영어 제외한 undamaged_documents에서 한글 데이터 있는 것과 없는 것 분리
    documents_with_korean = [
        document
        for document in tqdm(documents_without_english, desc="Process documents with Korean", mininterval=1)
        if document["korean"] is not None
    ]
    documents_without_korean = [
        document
        for document in tqdm(documents_without_english, desc="Process documents without Korean", mininterval=1)
        if document["korean"] is None
    ]

    # Hanja-English 데이터를 train, valid, test set으로 분리
    hanja_english_train_documents, hanja_english_test_documents = train_test_split(
        documents_with_english, test_size=args.translation_test_dataset_size, random_state=args.seed
    )
    hanja_english_train_documents, hanja_english_valid_documents = train_test_split(
        hanja_english_train_documents, test_size=args.translation_validation_dataset_size, random_state=args.seed
    )

    # Hanja-Korean 데이터를 train, valid, test set으로 분리
    hanja_korean_train_documents, hanja_korean_test_documents = train_test_split(
        documents_with_korean, test_size=args.translation_test_dataset_size, random_state=args.seed
    )
    hanja_korean_train_documents, hanja_korean_valid_documents = train_test_split(
        hanja_korean_train_documents, test_size=args.translation_validation_dataset_size, random_state=args.seed
    )

    # Hanja 데이터를 train, valid, test set으로 분리
    hanja_documents = documents_without_korean + hanja_english_train_documents + hanja_korean_train_documents

    hanja_train_documents, hanja_test_documents = train_test_split(
        hanja_documents, test_size=args.restoration_test_dataset_size, random_state=args.seed
    )
    hanja_train_documents, hanja_valid_documents = train_test_split(
        hanja_train_documents, test_size=args.restoration_valid_dataset_size, random_state=args.seed
    )

    logger.info(f"Total hanja train documents: {len(hanja_train_documents)}")
    logger.info(f"Total hanja valid documents: {len(hanja_valid_documents)}")
    logger.info(f"Total hanja test documents: {len(hanja_test_documents)}")
    logger.info(f"Total hanja-korean train documents: {len(hanja_korean_train_documents)}")
    logger.info(f"Total hanja-korean valid documents: {len(hanja_korean_valid_documents)}")
    logger.info(f"Total hanja-korean test documents: {len(hanja_korean_test_documents)}")
    logger.info(f"Total hanja-english train documents: {len(hanja_english_train_documents)}")
    logger.info(f"Total hanja-english valid documents: {len(hanja_english_valid_documents)}")
    logger.info(f"Total hanja-english test documents: {len(hanja_english_test_documents)}")

    # ner 단어 빈도 계산
    ner_freq = Counter()
    for document in tqdm(hanja_train_documents, desc="Calculate NER frequencies", mininterval=1):
        ner_freq.update(document["ner"])

    # N회 이상 출력한 NER 단어들의 개수 출력
    for f in [1, 5, 10, 20, 30, 50, 100]:
        logger.info(f"ner freq >= {f}: {len([freq for freq in ner_freq.values() if freq >= f])}")

    ner_vocab = [ner_text for ner_text, freq in ner_freq.items() if args.min_ner_vocab_freq <= freq]

    logger.info("Save files...")

    # restoration dataset
    with open(os.path.join(args.output_dir, "hanja_ner_vocab.txt"), "w") as f:
        f.write("\n".join(ner_vocab))

    with open(os.path.join(args.output_dir, "damaged_documents.json"), "w") as f:
        json.dump(damaged_documents, f, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "hanja_train_dataset.json"), "w") as f:
        json.dump(hanja_train_documents, f, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "hanja_valid_dataset.json"), "w") as f:
        json.dump(hanja_valid_documents, f, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "hanja_test_dataset.json"), "w") as f:
        json.dump(hanja_test_documents, f, ensure_ascii=False)

    # translation dataset
    with open(os.path.join(args.output_dir, "hanja_english_train_dataset.json"), "w") as f:
        json.dump(hanja_english_train_documents, f, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "hanja_english_valid_dataset.json"), "w") as f:
        json.dump(hanja_english_valid_documents, f, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "hanja_english_test_dataset.json"), "w") as f:
        json.dump(hanja_english_test_documents, f, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "hanja_korean_train_dataset.json"), "w") as f:
        json.dump(hanja_korean_train_documents, f, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "hanja_korean_valid_dataset.json"), "w") as f:
        json.dump(hanja_korean_valid_documents, f, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "hanja_korean_test_dataset.json"), "w") as f:
        json.dump(hanja_korean_test_documents, f, ensure_ascii=False)

    logger.info("File saved!")
