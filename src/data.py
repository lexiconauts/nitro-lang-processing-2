import torch
from torch.utils.data import Dataset
from torch import Tensor
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np
from transformers import BatchEncoding


from preprocess import TextPreprocessor, BERTPreprocessor, RobertaPreprocessor, MT5Preprocessor


class SexismDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, processor: TextPreprocessor[BatchEncoding]) -> None:
        super().__init__()

        # Initialization
        self.processor: TextPreprocessor[BatchEncoding] = processor

        # Keep a reference to the original dataset and the tokenized text
        self.dataset_pro, self.preprocessed_text = self.processor(
            self.dataset_pro)

        # Add labels to the dataset if they exist along with mappings
        self.__use_labels()

    def __getitem__(self, key: int | slice) -> Dict[str, torch.Tensor]:
        # Build output step by step
        output = dict()

        # Extract info needed for BERT - Should already be in tensor format
        input_ids, attention_mask = self.preprocessed_text[
            'input_ids'], self.preprocessed_text['attention_mask']
        output['attention_mask'] = attention_mask[key]
        output['input_ids'] = input_ids[key]

        # Add labels to the batch if it's a training set
        if self.has_labels:
            if isinstance(key, (int, np.int64, np.int32)):
                output['label'] = self.label_to_class[self.dataset_pro['label'][key]]
                output['label'] = torch.tensor(output['label'])
            else:
                output['label'] = self.dataset_pro['label'].iloc[key].map(
                    self.label_to_class)
                output['label'] = torch.tensor(output['label'].tolist())

        # Batched output
        return output

    def __len__(self) -> int:
        return len(self.dataset_pro)

    def __use_labels(self, ) -> None:
        # Check for label existance
        self.has_labels = 'label' in self.dataset_pro.columns

        if not self.has_labels:
            return

        # Create forward and backward mappings
        self.label_to_class: Dict[str, int] = {
            'direct': 0,
            'descriptive': 1,
            'reporting': 2,
            'non-offensive': 3,
            'offensive': 4,
        }
        self.class_to_label: Dict[int, str] = {
            v: k for k, v in self.label_to_class.items()}
        self.classes = np.vectorize(self.label_to_class.get)(
            self.dataset_pro['label'])
        self.freq_count = torch.from_numpy(
            np.unique(self.classes, return_counts=True)[1])
        self.weights = 1 - self.freq_count / self.freq_count.sum()
