import re
from random import Random
from typing import Dict, List, Optional, Tuple

import ahocorasick

from src.data_preprocessing.utils import DAMAGE_TOKEN

DN_TOKEN = re.compile(r"\[D\d+\]")
EXCLUDE_CHARS = {".", ",", ":", ";", " ", "·", "?", "'", '"', "(", ")", "{", "}", "<", ">"}
MASK_LENGTH_RATIO = [0.5, 0.3, 0.15, 0.03, 0.02]


def map_damage_tokens(
    masked_document: str, original_characters: str, damage_token: str = "□", mode: str = "dn"
) -> Tuple[str, Dict[str, str]]:
    damage_counter = 1
    out_chars, map_info = [], {}

    for char, orig in zip(masked_document, original_characters):
        if char == damage_token:
            key = f"[D{damage_counter}]"
            out_chars.append(key if mode == "dn" else "[MASK]")
            map_info[key] = orig
            damage_counter += 1
        else:
            out_chars.append(char)

    return "".join(out_chars), map_info


class MaskConverter:
    def __init__(
        self,
        average_mask_ratio: float = 0.15,
        max_mask_ratio: float = 0.3,
        ner_vocab: Optional[List[str]] = None,
        max_num_masking_iterations: int = 50,
        seed: int = 42,
        mode: str = "dn",
        masking_strategy: str = "random",
    ):
        self.average_mask_ratio = average_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.max_num_masking_iterations = max_num_masking_iterations
        self.rng = Random(seed)
        self.mode = mode
        self.masking_strategy = masking_strategy

        self.ner_vocab = None
        if ner_vocab:
            automaton = ahocorasick.Automaton()
            for idx, word in enumerate(ner_vocab):
                automaton.add_word(word, (idx, word))

            automaton.make_automaton()
            self.ner_vocab = automaton

    def _collect_ner_spans(self, text: str) -> List[Tuple[int, int, str]]:
        spans = []
        if self.ner_vocab is None:
            return spans

        for end, (_, w) in self.ner_vocab.iter(text):
            start = end - len(w) + 1
            spans.append((start, end, w))
        return spans

    def _collect_spans_from_terms(self, text: str, terms: List[str]) -> List[Tuple[int, int, str]]:
        spans: List[Tuple[int, int, str]] = []
        if not terms:
            return spans
        auto = ahocorasick.Automaton()
        for i, w in enumerate(terms):
            if not w:
                continue
            auto.add_word(w, (i, w))
        auto.make_automaton()
        for end, (_, w) in auto.iter(text):
            start = end - len(w) + 1
            spans.append((start, end, w))
        return spans

    def _mask_span_chars(self, masked_document: List[str], start: int, end: int) -> int:
        added = 0
        for i in range(start, end + 1):
            if masked_document[i] in EXCLUDE_CHARS or masked_document[i] == DAMAGE_TOKEN:
                continue
            masked_document[i] = DAMAGE_TOKEN
            added += 1
        return added

    def _random_ngram_masking(self, masked_document: List[str], original: str, target: int, already: int) -> int:
        total = already
        num_trials = 0
        while total < target and num_trials < self.max_num_masking_iterations:
            num_trials += 1
            mask_len = self.rng.choices([i + 1 for i in range(len(MASK_LENGTH_RATIO))], weights=MASK_LENGTH_RATIO)[0]
            start = self.rng.randint(0, max(0, len(original) - mask_len))
            # 문장부호/공백 제외, 인접 손상과 겹치면 skip
            if any(masked_document[i] in EXCLUDE_CHARS for i in range(start, start + mask_len)) or any(
                ch == DAMAGE_TOKEN or ch == " " for ch in masked_document[max(start - 1, 0) : start + mask_len + 1]
            ):
                continue
            for i in range(start, start + mask_len):
                masked_document[i] = DAMAGE_TOKEN
            total += mask_len
        return total

    def mask_document(
        self, original_document: str, sample_ner: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, str]]:
        masked_document = list(original_document)

        # masking strategy가 ner_only 인 경우
        if self.masking_strategy == "ner_only":
            if not sample_ner:
                raise ValueError("No per-sample NER; skipping in ner_only.")

            spans = self._collect_spans_from_terms(original_document, sample_ner)
            if not spans:
                raise ValueError("No NER matches for ner_only")

            mask_ratio = max(self.rng.gauss(self.average_mask_ratio * 100) / 100, 0.0)
            mask_ratio = min(mask_ratio, self.max_mask_ratio)
            target = max(round(len(original_document) * mask_ratio), 1)
            self.rng.shuffle(spans)

            total = 0
            for start, end, _ in spans:
                if total >= target:
                    break
                total += self._mask_span_chars(masked_document, start, end)

            masked_text, mask_label = map_damage_tokens(
                "".join(masked_document), original_document, damage_token=DAMAGE_TOKEN, mode=self.mode
            )
            return masked_text, mask_label

        if self.masking_strategy == "random" or self.masking_strategy == "mixed":
            # masking strategy가 mixed 혹은 random 인 경우
            # average_mask_ratio를 기준으로 정규 분포에서 샘플링하여 mask_ratio 설정
            mask_ratio = max(self.rng.gauss(self.average_mask_ratio * 100) / 100, 0.0)
            # mask_ratio가 max_mask_ratio를 넘지 않도록 설정
            mask_ratio = min(mask_ratio, self.max_mask_ratio)
            # 원본 문서 길이에 mask_ratio를 곱해 masking할 토큰 수를 결정
            # 최소한 1개 이상의 토큰이 masking되도록 설정
            target = max(round(len(original_document) * mask_ratio), 1)

            total = 0
            if self.masking_strategy == "mixed" and self.ner_vocab is not None:
                spans = self._collect_ner_spans(original_document)
                self.rng.shuffle(spans)

                for start, end, _ in spans:
                    if total >= target:
                        break
                    total += self._mask_span_chars(masked_document, start, end)

            # random_only 또는 mixed의 잔여량을 n-gram 랜덤으로 채우기
            total = self._random_ngram_masking(masked_document, original_document, target, total)

            # 하나도 마스킹이 되지 않은 경우 랜덤하게 1개 마스킹해주기
            if total == 0:
                idx = self.rng.randint(0, len(original_document) - 1)
                masked_document[idx] = DAMAGE_TOKEN

            masked_text, mask_label = map_damage_tokens(
                "".join(masked_document), original_document, damage_token=DAMAGE_TOKEN, mode=self.mode
            )
            return masked_text, mask_label
