import math
from typing import Iterator, List, TypeVar

import numpy as np
import torch
from torch.utils.data import Sampler

T_co = TypeVar("T_co", covariant=True)


class CollateFn:
    def __init__(self, pad_token_id: int, label_mask_id: int, max_length: int):
        self.pad_token_id = pad_token_id
        self.label_mask_id = label_mask_id
        self.max_length = max_length

    def __call__(self, batch):
        input_ids, label_ids = zip(*batch)
        max_len = min(max(len(ids) for ids in input_ids), self.max_length)

        batched_input_ids = torch.full((len(input_ids), max_len), self.pad_token_id, dtype=torch.long)
        batched_label_ids = torch.full((len(label_ids), max_len), self.label_mask_id, dtype=torch.long)

        for i, (input_id, label_id) in enumerate(zip(input_ids, label_ids)):
            input_length = len(input_id)
            batched_input_ids[i, :input_length] = input_id
            batched_label_ids[i, :input_length] = label_id

        attention_mask = (batched_input_ids != self.pad_token_id).long()

        return {
            "input_ids": batched_input_ids,
            "labels": batched_label_ids,
            "attention_mask": attention_mask,
        }


class LengthBasedBatchSampler(Sampler[T_co]):
    # 비슷한 길이끼리 배치를 묶어주는 Distributed Sampler
    # Distributed Sampler는 아래 토치 원 코드 참조
    # https://github.com/pytorch/pytorch/blob/39452c7b5bd47858a7ac087107246fe19bdf489b/torch/utils/data/distributed.py#L17
    def __init__(
        self,
        data_lengths: List[int],
        num_replicas: int,
        rank: int,
        batch_size_per_device: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.data_lengths = data_lengths
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size_per_device = batch_size_per_device
        self.total_batch_size = self.batch_size_per_device * self.num_replicas
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.data_lengths) % self.total_batch_size != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = (
                math.ceil((len(self.data_lengths) - self.total_batch_size) / self.total_batch_size)
                * self.batch_size_per_device
            )
        else:
            self.num_samples = math.ceil(len(self.data_lengths) / self.total_batch_size) * self.batch_size_per_device

        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        indices = np.array(self.data_lengths).argsort().tolist()
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]

        # shape을 (iteration, num_replicas * batch_size)로 변경
        indices = np.array(indices)
        indices = indices.reshape(-1, self.total_batch_size)

        longest_batch = indices[-1, :]
        rest_batch = indices[:-1, :]

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            rng = np.random.default_rng(self.seed + self.epoch)
            # 긴 길이 배치에 대해서 셔플
            rng.shuffle(rest_batch[0, :])
            # 전체 배치 내, 배치 간 순서를 섞음
            rng.shuffle(rest_batch)
            for i in range(len(rest_batch)):
                rng.shuffle(rest_batch[i])

        indices = np.concatenate([longest_batch.reshape(1, -1), rest_batch], axis=0)

        # 다시 shape을 (iteration * num_replicas * batch_size, )로 변경
        indices = indices.reshape(-1).tolist()

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
