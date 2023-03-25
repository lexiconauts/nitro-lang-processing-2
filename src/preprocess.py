from abc import ABC, abstractmethod
import pandas as pd
from torch import Tensor
from pandas import Series, DataFrame
from transformers import AutoTokenizer, BatchEncoding
import typing
from typing import List, Set, Dict, TypeVar, Generic


PT = TypeVar('PT')

url_regex = "https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
twitter_handle_regex = '@[a-zA-Z0-9_-]+'


def remove_by_regex(df: DataFrame, column: str, regex: str):
    df[column] = df[column].str.replace(regex, '', regex=True)


class TextPreprocessor(ABC, Generic[PT]):
    def __init__(self, dataset: DataFrame) -> None:
        self.dataset = dataset

    @abstractmethod
    def __call__(dataset: DataFrame) -> PT:
        raise NotImplemented()


class BertPreprocessor(TextPreprocessor[BatchEncoding]):
    def __init__(self, dataset: DataFrame, repo: str) -> None:
        super().__init__(dataset=dataset)

        # 94 chrs, media 120chr - 20 mean toks
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.max_length = 32  # TODO: adjust this?

    def __call__(self, dataset: DataFrame) -> BatchEncoding:
        # Manual preprocessing
        sentences = []
        for text in dataset['text'].tolist():
            text = text \
                .replace("ţ", "ț") \
                .replace("ş", "ș") \
                .replace("Ţ", "Ț") \
                .replace("Ş", "Ș")
            sentences.append(text)

        # Retrieve max-length
        return self.tokenizer.__call__(
            sentences,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )


class RobertaPreprocessor(TextPreprocessor[BatchEncoding]):
    def __init__(self, dataset: DataFrame, repo: str) -> None:
        super().__init__(dataset=dataset)

        # 94 chrs, media 120chr - 20 mean toks
        self.tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=True)
        self.max_length = 256  # TODO: adjust this?

    def __call__(self, dataset: DataFrame) -> BatchEncoding:
        # Manual preprocessing
        sentences = []
        for text in dataset['text'].tolist():
            text = text \
                .replace("ţ", "ț") \
                .replace("ş", "ș") \
                .replace("Ţ", "Ț") \
                .replace("Ş", "Ș")
            sentences.append(text)

        # Retrieve max-length
        return self.tokenizer.__call__(
            sentences,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
