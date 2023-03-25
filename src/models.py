import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel
from typing import Dict


class BertFlatClassModel(nn.Module):
    def __init__(self, repo: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Init the base model
        self.model = AutoModel.from_pretrained(repo)
        self.modify_structure()
        self.unfreeze()

    def modify_structure(self) -> None:
        pass

    def unfreeze(self) -> None:
        pass

    def forward(self, x: Dict[str, Tensor]):
        pass

