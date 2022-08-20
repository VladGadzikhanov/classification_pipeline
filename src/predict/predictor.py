import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    outputs = []
    tags = None
    for i_b, batch in enumerate(tqdm(loader)):
        tags, output = model.one_shot_predict(batch, device)
        outputs.append(output)
    return pd.DataFrame(data=np.concatenate(outputs), columns=tags)
