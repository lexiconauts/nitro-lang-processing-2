import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel
from transformers import BertModel
from typing import Dict, List, Set, Tuple, Literal


class BertFlatClassModel(nn.Module):
    def __init__(self,
                 repo: str,
                 unfreeze: Literal['all'] | Literal['none'] | List[str] = 'none',
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Init the base model
        self.n_classes = 5
        self.bert_model: BertModel = AutoModel.from_pretrained(repo)
        self.unfreeze(unfreeze)
        self.create_layers()

    def create_layers(self) -> None:
        self.layers = nn.Sequential(
            nn.Linear(in_features=768, out_features=self.n_classes)
        )

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

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        # Extract the relevant data
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']

        # Call the pretrained model
        _, output = self.bert_model(input_ids, attention_mask)

        # Add layers over it
        output = self.layers(output)

        return output

