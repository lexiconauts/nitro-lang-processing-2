from abc import ABC, abstractmethod
import pandas as pd
from torch import Tensor
from pandas import Series, DataFrame
from transformers import AutoTokenizer, BatchEncoding, T5Tokenizer
import typing
from typing import List, Set, Dict, TypeVar, Generic, Optional, Tuple


PT = TypeVar('PT')

URL_REGEX = "https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
TWITTER_HANDLE_REGEX = '@[a-zA-Z0-9_-]+'
NUMBER_REGEX = r"\d+"


class TextPreprocessor(ABC, Generic[PT]):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(dataset: DataFrame) -> PT:
        raise NotImplemented()


class AutoPreprocessor(TextPreprocessor[BatchEncoding]):
    def __init__(self, repo: str, max_length: int = 64) -> None:
        super().__init__()

        # Initializer internals
        self.repo: str = repo
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo)
        self.max_length: int = max_length

    def __call__(self, dataset: DataFrame) -> Tuple[DataFrame, BatchEncoding]:
        # Remove bad examples
        dataset = dataset.drop(
            dataset[dataset['text'].str.contains('\t')].index)
        dataset = dataset.reset_index()

        # Remove twitter handles
        dataset['text'] = dataset['text'].str.replace(URL_REGEX, ' ', regex=True)
        dataset['text'] = dataset['text'].str.replace(TWITTER_HANDLE_REGEX, ' ', regex=True)
        dataset['text'] = dataset['text'].str.replace(NUMBER_REGEX, ' ', regex=True)

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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(repo='dumitrescustefan/bert-base-romanian-cased-v1',
                         *args, **kwargs)


class MT5Preprocessor(AutoPreprocessor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(repo='dumitrescustefan/mt5-large-romanian',
                         *args, **kwargs)


class RobertPreprocessor(AutoPreprocessor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(repo='readerbench/RoBERT-base',
                         *args, **kwargs)


def remove_by_regex(df: DataFrame, column: str, regex: str) -> None:
    df[column] = df[column].str.replace(regex, '', regex=True)
