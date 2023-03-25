import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import GELU
from transformers import AutoModel, BertModel, MT5Model, T5Tokenizer
from typing import Dict, List, Set, Tuple, Literal
import abc
from abc import ABC, abstractmethod
import numpy as np


class Args(object):
    def __init__(self) -> None:
        # Training specifications
        self.num_epochs = 30
        self.num_workers = 1
        self.batch_size = 32
        self.weight_decay = 5e-4
        self.learning_rate = 1e-5


class Output(object):
    def __init__(self, predictions: List[int], loss_seq: List[float], accy_seq: List[float], class_to_label: Dict[int, str]) -> None:
        self.predictions_as_indx: np.ndarray = np.array(predictions)
        self.predictions_as_text: np.ndarray = np.vectorize(class_to_label.get)(self.predictions_as_indx)
        self.loss_seq: np.ndarray = np.array(loss_seq)
        self.accy_seq: np.ndarray = np.array(accy_seq)
        self.loss_mean: float = self.loss_seq.mean()
        self.accy_mean: float = self.accy_seq.mean()


class PretrainedFlatClassModel(ABC, nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def create_layers(self) -> nn.Sequential:
        raise NotImplementedError()

    def unfreeze(self, layers: str | List[str], unfreeze: bool = True) -> None:
        if layers == 'all':
            for param in self.parameters():
                param.requires_grad = unfreeze
        elif layers == 'none':
            self.requires_grad_(False)
            return
        else:
            for (param_name, param) in self.named_parameters():
                if param_name in layers:
                    param.requires_grad = unfreeze


class BertFlatClassModel(PretrainedFlatClassModel):
    def __init__(self,
                 dropout: float = 0.1,
                 unfreeze: Literal['all'] | Literal['none'] | List[str] = 'none',
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Save internal params
        self.dropout: float = dropout

        # Init the base model
        self.n_classes = 5
        self.repo = 'dumitrescustefan/bert-base-romanian-cased-v1'
        self.bert_model: BertModel = AutoModel.from_pretrained(self.repo)
        self.unfreeze(unfreeze)
        self.create_layers()

    def create_layers(self) -> None:
        self.layers = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=768, out_features=768),
            nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=768, out_features=self.n_classes),
        )

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        # Extract the relevant data
        input_ids: Tensor = x['input_ids']
        attention_mask: Tensor = x['attention_mask']

        # Call the pretrained model
        _, output = self.bert_model(input_ids, attention_mask, return_dict=False)

        # Add layers over it
        output = self.layers(output)

        return output

