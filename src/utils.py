import torch
from typing import Tuple
import pathlib as pb
import pandas as pd

def get_available_device() -> torch.device:
    # Use available GPU
    device: torch.device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


def read_data(data_dir: pb.Path, train_filename: str = 'train_data.csv', test_filename: str = 'test_data.csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Read CSVs
    train_data_raw = pd.read_csv(data_dir / train_filename, sep=',')
    test_data_raw = pd.read_csv(data_dir / test_filename, sep=',')

    # Adjust column headers
    train_data_raw = train_data_raw.rename(columns={ 'Text': 'text', 'Final Labels': 'label' })
    train_data_raw = train_data_raw.drop(columns='Id', errors='ignore')
    test_data_raw = test_data_raw.rename(columns={ 'Text': 'text' })
    test_data_raw = test_data_raw.drop(columns='Id', errors='ignore')

    return train_data_raw, test_data_raw

