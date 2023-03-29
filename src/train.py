import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import List, Dict, Callable, Tuple, Optional
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from models import Output, Args
from data import SexismDataset


def train(model: nn.Module, optimizer: Optimizer, loss_fn: nn.Module, data_loader: DataLoader, class_to_label: Dict[int, str], args: Args, device: torch.device = torch.device('cpu'), valid_callback: Optional[Callable[[], None]] = None) -> Output:
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss: List[float] = []
        epoch_accy: List[float] = []
        predictions: List[int] = []

        for batch_i, batch in enumerate(data_loader):
            # Send batch to GPU
            batch: Dict[str, Tensor] = { k: v.to(device) for k, v in batch.items() }

            # Make predictions
            y_true: Tensor | np.ndarray = batch['label']
            y_pred: Tensor | np.ndarray = model.forward(batch)

            # Compute the loss
            optimizer.zero_grad()
            loss: Tensor = loss_fn(y_pred, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Compute the accuracy
            y_true = y_true.detach().cpu().numpy()
            y_pred = y_pred.detach().argmax(dim=1).cpu().numpy()
            predictions.extend(y_pred.tolist())
            epoch_accy.append(balanced_accuracy_score(y_true, y_pred))

            # Track progress
            epoch_loss.append(loss.detach().cpu().numpy())
        
        # Track progress over each epoch
        output = Output(predictions, epoch_loss, epoch_accy, class_to_label=class_to_label, with_labels=True)
        print('Train Epoch {} - Loss: {}, Accuracy: {}'.format(epoch, output.loss_mean, output.accy_mean))

        # Call Evaluation overValidation
        if valid_callback:
            valid_callback()

    # Retain and expose output
    return output


def evaluate(model: nn.Module, loss_fn: nn.Module, data_loader: DataLoader, class_to_label: Dict[int, str], with_labels: bool = True, device: torch.device = torch.device('cpu')) -> Output:
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
    output = Output(predictions, epoch_loss, epoch_accy, class_to_label=class_to_label, with_labels=with_labels)
    return output

