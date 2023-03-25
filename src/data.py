import torch
from torch.utils.data import Dataset
from torch import Tensor
from typing import Dict, List, Tuple, Set
import pandas as pd
from transformers import BatchEncoding


from preprocess import TextPreprocessor, BertPreprocessor


class SexismDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, processor: TextPreprocessor[BatchEncoding]) -> None:
        super().__init__()
        self.processor = processor
        self.dataset_raw = dataset
        self.preprocessed_text = self.preprocess()
        self.has_labels = 'label' in dataset.columns
        self.label_to_class: Dict[str, int] = {
            'direct': 0,
            'descriptive': 1,
            'reporting': 2,
            'non-offensive': 3,
            'offensive': 4,
        }

    def preprocess(self) -> BatchEncoding:
        return self.processor(self.dataset_raw)

    def __getitem__(self, key: int | slice) -> Dict[str, torch.Tensor]:
        # Build output step by step
        output = dict()

        # Extract info needed for BERT - Should already be in tensor format
        input_ids, attention_mask = self.preprocessed_text['input_ids'], self.preprocessed_text['attention_mask']
        output['attention_mask'] = attention_mask[key]
        output['input_ids'] = input_ids[key]

        # Add optional labels
        if self.has_labels:
            if isinstance(key, int):
                output['label'] = self.label_to_class[self.dataset_raw['label'][key]]
            else:
                output['label'] = self.dataset_raw['label'][key].map(self.label_to_class)

            # Transform to tensor
            output['label'] = torch.tensor(output['label'].tolist())

        # Batched output
        return output

    def __len__(self) -> int:
        return len(self.dataset_raw)

