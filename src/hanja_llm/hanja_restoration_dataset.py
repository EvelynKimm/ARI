from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.hanja_llm.prompt import SYSTEM_PROMPT, USER_PROMPT

LABEL_MASK_ID = -100


class HanjaRestorationDatasetForTrain(Dataset):
    def __init__(self, examples: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int = 4096):

        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        rationale = example.get("rationale")
        masked_document, mask_label = example["masked_document"], example["mask_label"]

        if rationale:
            enable_thinking = True
        else:
            enable_thinking = False

        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    example["related_documents_fuzz80_top20"],
                    example["date"],
                    example["month"],
                    example["year"],
                    example["king"],
                    masked_document,
                ),
            },
        ]

        context_ids = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False, add_special_tokens=False, enable_thinking=enable_thinking
        )

        context_ids = self.tokenizer.encode(context_ids)

        if rationale:
            response_id = self.tokenizer.encode(rationale) + self.tokenizer.encode(mask_label)
        else:
            response_id = self.tokenizer.encode(mask_label)

        eos_id = self.tokenizer.eos_token_id
        input_ids = context_ids + response_id + [eos_id]
        label_ids = [LABEL_MASK_ID for i in range(len(context_ids))] + response_id + [eos_id]

        if len(input_ids) > self.max_length:
            print(f"[WARN] sample idx={idx}: len(input_ids)={len(input_ids)} > max_length={self.max_length}")
        
        input_ids = input_ids[: self.max_length]
        label_ids = label_ids[: self.max_length]

        return torch.tensor(input_ids), torch.tensor(label_ids)
