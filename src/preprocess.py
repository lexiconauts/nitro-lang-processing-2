from abc import ABC, abstractmethod
import pandas as pd
from torch import Tensor
from pandas import Series, DataFrame
from transformers import AutoTokenizer, BatchEncoding, T5Tokenizer
import typing
from typing import List, Set, Dict, TypeVar, Generic, Optional


PT = TypeVar('PT')

URL_REGEX = "https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
TWITTER_HANDLE_REGEX = '@[a-zA-Z0-9_-]+'


class TextPreprocessor(ABC, Generic[PT]):
    def __init__(self, dataset: Optional[DataFrame] = None) -> None:
        self.dataset = dataset

    @abstractmethod
    def __call__(dataset: DataFrame) -> PT:
        raise NotImplemented()


class AutoPreprocessor(TextPreprocessor[BatchEncoding]):
    def __init__(self, repo: str, max_length: int = 32, dataset: Optional[DataFrame] = None) -> None:
        super().__init__(dataset)

        # Initializer internals
        self.repo: str = repo
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo)
        self.max_length: int = max_length

    def __call__(self, dataset: DataFrame) -> (DataFrame, BatchEncoding):
        # Remove bad examples
        dataset = dataset.drop(
            dataset[dataset['text'].str.contains('\t')].index)

        # Manual preprocessing
        sentences: List[str] = []
        for text in dataset['text'].tolist():
            text: str = text \
                .replace("ţ", "ț") \
                .replace("ş", "ș") \
                .replace("Ţ", "Ț") \
                .replace("Ş", "Ș")
            sentences.append(text)

        # Retrieve max-length
        return dataset, self.tokenizer.__call__(
            sentences,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )


class BERTPreprocessor(AutoPreprocessor):
    def __init__(self, dataset: Optional[DataFrame] = None, *args, **kwargs) -> None:
        super().__init__(dataset=dataset,
                         repo='dumitrescustefan/bert-base-romanian-cased-v1',
                         *args, **kwargs)


class MT5Preprocessor(BERTPreprocessor):
    def __init__(self, dataset: Optional[DataFrame] = None, *args, **kwargs) -> None:
        super().__init__(dataset,
                         repo='dumitrescustefan/mt5-large-romanian',
                         *args, **kwargs)


class RobertaPreprocessor(BERTPreprocessor):
    def __init__(self, dataset: Optional[DataFrame] = None, *args, **kwargs) -> None:
        super().__init__(dataset=dataset,
                         repo='readerbench/RoBERT-base',
                         *args, **kwargs)


def remove_by_regex(df: DataFrame, column: str, regex: str) -> None:
    df[column] = df[column].str.replace(regex, '', regex=True)
