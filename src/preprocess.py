from abc import ABC, abstractmethod
import pandas as pd
from torch import Tensor
from pandas import Series, DataFrame
from transformers import AutoTokenizer, BatchEncoding
import typing
from typing import List, Set, Dict, TypeVar, Generic


PT = TypeVar('PT')


class TextPreprocessor(ABC, Generic[PT]):
    def __init__(self, dataset: DataFrame) -> None:
        self.dataset = dataset

    @abstractmethod
    def __call__(dataset: DataFrame) -> PT:
        raise NotImplemented()


class BertPreprocessor(TextPreprocessor[BatchEncoding]):
    def __init__(self, dataset: DataFrame, repo: str) -> None:
        super().__init__(dataset=dataset)
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.max_length = 256 # TODO: adjust this?

    def __call__(self, dataset: DataFrame) -> BatchEncoding:
        # Retrieve max-length
        return self.tokenizer.__call__(
            dataset['text'].tolist(),
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
