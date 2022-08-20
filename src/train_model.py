import os
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from src.utils.pipeline import load_model, set_device
from src.types import TrainerType, CollectionType
from torch.optim import Optimizer


def train_model(
        Model: torch.nn.Module,
        Trainer: TrainerType,
        Collection: CollectionType,
        criterion: Callable,
        optimizer: Optimizer,
        config: dict,
        experiment_folder_path: str,
        epoch_to_load=None
) -> int:

    print("Starting model training stage...")
    experiment_params = config['experiment_param']
    train_params = config['train_param']
    model_params = Model.model_params

    experiment_name = experiment_params["name"]
    path_tensorboard = experiment_params["path_tensorboard"]

    device = set_device(train_params["cuda_number"])
    use_balanced_weights = train_params["use_balanced_weights"]
    tb_path = os.path.join(path_tensorboard, experiment_name)

    # for densenet
    if "growth_rate" in train_params:
        train_params["num_init_features"] = train_params["growth_rate"] * 2

    train = pd.read_csv(os.path.join(experiment_folder_path, "train.csv"))
    print("Train set size: ", train.shape)
    val = pd.read_csv(os.path.join(experiment_folder_path, "val.csv"))
    print("Val set size: ", val.shape)

    train_c = Collection(train, "train")
    val_c = Collection(val, "val")

    batch_size = train_params["batch_size"]
    num_workers = train_params["num_workers"]

    data_loader = DataLoader(train_c, batch_size=batch_size, num_workers=num_workers, pin_memory=False, shuffle=True)
    val_loader = DataLoader(val_c, batch_size=batch_size, num_workers=num_workers, pin_memory=False, shuffle=False)

    if use_balanced_weights:
        weights = compute_class_weight("balanced", classes=np.arange(model_params["num_classes"]), y=train["target"])
        print("Class weights: \n", weights)
        weights = torch.tensor(weights).to(device, torch.float)
    else:
        weights = None

    criterion = criterion(weight=weights)

    lr = train_params["lr"]
    num_epochs = train_params["num_epochs"]

    if epoch_to_load is not None:
        model_path = os.path.join(experiment_folder_path, "model", f"best_model_{epoch_to_load}.pt")
        Model = load_model(model_path, Model, device)

    trainer = Trainer(
        Model, criterion, optimizer, experiment_folder_path,
        tb_path=tb_path, lr=lr, device=device, num_epochs=num_epochs, epoch_to_load=epoch_to_load)
    epoch = trainer.run(data_loader, val_loader)
    print("Model training is finished.")
    return epoch
