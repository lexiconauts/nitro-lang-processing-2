from abc import ABC, abstractmethod
import pandas as pd
from pandas import Series, DataFrame
from transformers import AutoTokenizer
from tokenizers import Pre

class TextPreprocessor(ABC):
    def __init__(self, dataset: DataFrame) -> None:
        self.dataset = dataset

    @abstractmethod
    def preprocess(text: str) -> str:
        raise NotImplemented()


class BertPreprocessor(TextPreprocessor):
    def __init__(self, dataset: DataFrame, repo: str) -> None:
        super().__init__(dataset=dataset)
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.max_length = 256 # TODO: adjust this?
        self.preprocessed = False

    def preprocess(self, text: str) -> str:
        # Retrieve max-length
        self.tokenizer.__call__(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
        )
