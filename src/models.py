import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import GELU
from transformers import AutoModel, BertModel, MT5Model, MT5EncoderModel, T5Tokenizer
from typing import Dict, List, Set, Tuple, Literal
import abc
from abc import ABC, abstractmethod
import numpy as np


class Args(object):
    def __init__(self) -> None:
        # Training specifications
        self.num_epochs = 3
        self.num_workers = 1
        self.batch_size = 32
        self.weight_decay = 5e-6
        self.learning_rate = 2e-5


class Output(object):
    def __init__(self, predictions: List[int], loss_seq: List[float], accy_seq: List[float], class_to_label: Dict[int, str], with_labels: bool) -> None:
        self.predictions_as_indx: np.ndarray = np.array(predictions)
        self.predictions_as_text: np.ndarray = np.vectorize(
            class_to_label.get)(self.predictions_as_indx)
        self.with_labels: bool = with_labels
        if self.with_labels:
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

    def unfreeze(self, layers: str | List[str] | List[nn.Module], unfreeze: bool = True) -> None:
        if isinstance(layers, list) and isinstance(layers[0], nn.Module):
            for module in layers:
                for param in module.parameters():
                    param.requires_grad = unfreeze
        elif layers == 'all':
            for param in self.parameters():
                param.requires_grad = unfreeze
        elif layers == 'none':
            self.requires_grad_(False)
            return
        elif isinstance(layers[0], str):
            for (param_name, param) in self.named_parameters():
                if param_name in layers:
                    param.requires_grad = unfreeze
        else:
            raise ValueError('invalid layers param - bad unfreeze')


class MT5FlatClassModel(PretrainedFlatClassModel):
    def __init__(self, unfreeze: Literal['all'] | Literal['none'] | List[str] = 'none', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.fc_head = torch.nn.Linear(40 * 768, 5)
        self.mt5: MT5Model = MT5EncoderModel.from_pretrained(
            "dumitrescustefan/mt5-base-romanian")

        self.unfreeze([
            self.mt5.encoder.block[10].layer[0].SelfAttention.q.weight,
            self.mt5.encoder.block[10].layer[0].SelfAttention.k.weight,
            self.mt5.encoder.block[10].layer[0].SelfAttention.v.weight,
            self.mt5.encoder.block[10].layer[0].SelfAttention.o.weight,
            self.mt5.encoder.block[10].layer[0].layer_norm.weight,
            self.mt5.encoder.block[10].layer[1].DenseReluDense.wi_0.weight,
            self.mt5.encoder.block[10].layer[1].DenseReluDense.wi_1.weight,
            self.mt5.encoder.block[10].layer[1].DenseReluDense.wo.weight,
            self.mt5.encoder.block[10].layer[1].layer_norm.weight,
            self.mt5.encoder.block[11].layer[0].SelfAttention.q.weight,
            self.mt5.encoder.block[11].layer[0].SelfAttention.k.weight,
            self.mt5.encoder.block[11].layer[0].SelfAttention.v.weight,
            self.mt5.encoder.block[11].layer[0].SelfAttention.o.weight,
            self.mt5.encoder.block[11].layer[0].layer_norm.weight,
            self.mt5.encoder.block[11].layer[1].DenseReluDense.wi_0.weight,
            self.mt5.encoder.block[11].layer[1].DenseReluDense.wi_1.weight,
            self.mt5.encoder.block[11].layer[1].DenseReluDense.wo.weight,
            self.mt5.encoder.block[11].layer[1].layer_norm.weight,
            self.mt5.encoder.final_layer_norm.weight
        ])

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        y = self.mt5(
            input_ids=x['input_ids'], attention_mask=x['attention_mask']).last_hidden_state
        y = y.view(y.shape[0], -1)
        y = self.fc_head(y)
        return y


class BertFlatClassModel(PretrainedFlatClassModel):
    def __init__(self,
                 dropout: float = 0.1,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Save internal params
        self.dropout: float = dropout

        # Init the base model
        self.n_classes = 5
        self.repo = 'dumitrescustefan/bert-base-romanian-cased-v1'
        self.bert_model: BertModel = AutoModel.from_pretrained(self.repo)
        self.unfreeze([
            self.bert_model.pooler.dense,
            self.bert_model.encoder.layer[-1:]
        ])
        self.create_layers()

    def create_layers(self) -> None:
        self.layers = nn.Sequential(
            nn.Linear(in_features=768, out_features=512),
            nn.GELU(),
            nn.Linear(in_features=512, out_features=self.n_classes)
        )

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        # Extract the relevant data
        input_ids: Tensor = x['input_ids']
        attention_mask: Tensor = x['attention_mask']

        # Call the pretrained model
        _, output = self.bert_model(
            input_ids, attention_mask, return_dict=False)

        # Add layers over it
        output = self.layers(output)

        return output


class RoBertFlatClassModel(PretrainedFlatClassModel):
    def __init__(self,
                 dropout: float = 0.1,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Save internal params
        self.dropout: float = dropout

        # Init the base model
        self.n_classes = 5
        self.repo = 'readerbench/RoBERT-base'
        self.bert_model: BertModel = AutoModel.from_pretrained(self.repo)
        self.unfreeze([
            self.bert_model.pooler.dense,
            self.bert_model.encoder.layer[-1:]
        ])
        self.create_layers()

    def create_layers(self) -> None:
        self.layers = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_features=768, out_features=self.n_classes)
        )

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        # Extract the relevant data
        input_ids: Tensor = x['input_ids']
        attention_mask: Tensor = x['attention_mask']

        # Call the pretrained model
        _, output = self.bert_model(
            input_ids, attention_mask, return_dict=False)

        # Add layers over it
        output = self.layers(output)

        return output


class Ensemble(nn.Module):
    def __init__(self, models: List[nn.Module], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Init models
        self.models = models

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        y_hats = []  # 32xMx5

        for model in self.models:
            y_hat = model.forward(x).detach().cpu()
            y_hats.append(y_hat)

        stacked = torch.stack(y_hats, dim=1)
        predictions = torch.sum(stacked, dim=1)
        return predictions  # , torch.softmax(predictions, dim=1)
