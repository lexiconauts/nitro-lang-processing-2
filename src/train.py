import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from typing import List, Dict
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from models import Output
from data import SexismDataset


def evaluate(model: nn.Module, loss_fn: nn.Module, data_loader: DataLoader, with_labels: bool = True, device: torch.device = torch.device('cpu')) -> Output:
    model.eval()
    with torch.no_grad():
        epoch_loss: List[float] = []
        epoch_accy: List[float] = []
        predictions: List[int] = []
        
        # Single pass through the data
        for _, batch in enumerate(data_loader):
            # Send batch to GPU
            batch: Dict[str, Tensor] = { k: v.to(device) for k, v in batch.items() }

            # Make predictions
            y_pred: Tensor | np.ndarray = model.forward(batch)

            # Usable for both validation and testing
            if with_labels:
                # Compute the loss
                y_true: Tensor | np.ndarray = batch['label']
                loss: Tensor = loss_fn(y_pred, y_true)
            
            # Retain the predictions
            y_pred = y_pred.detach().argmax(dim=1).cpu().numpy()
            predictions.extend(y_pred.tolist())

            # Track metrics
            if with_labels:
                y_true = y_true.detach().cpu().numpy()
                epoch_accy.append(balanced_accuracy_score(y_true, y_pred))
                epoch_loss.append(loss.detach().cpu().numpy())

    # Retain and expose output
    dataset: SexismDataset = data_loader.dataset
    output = Output(predictions, epoch_loss, epoch_accy, class_to_label=dataset.class_to_label)
    return output

