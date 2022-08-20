import os

import pandas as pd
import torch

from src.predict import predict
from src.utils.general import load_yaml, save_json
from src.utils.metrics import compute_metrics
from src.utils.pipeline import build_data_loader, load_model, set_device, prepare_model_for_eval
from src.types import CollectionType


def _evaluate(model, data_loader, data_set, device, class_names=None):
    pred_df = predict(model, data_loader, device)
    data_set['obj_id'] = data_set['obj_id'].astype(object)
    df = pd.merge(pred_df, data_set, on=["img_path", "obj_id"], how="left")
    # cast to numeric columns if possible
    for s in df.columns:
        try:
            df[s] = pd.to_numeric(df[s])
        except ValueError:
            pass
        except TypeError:
            pass
    # calculate metrics
    result = compute_metrics(df, class_names)
    return df, result


def evaluate_model(
        Model: torch.nn.Module,
        Collection: CollectionType,
        experiment_folder_path: str,
        evaluation_set_name: str,
        config: dict,
        epoch_to_load: int = None,
        save_results: bool = True
):
    data_path = evaluation_set_name + ".csv"
    train_params = config['train_param']

    device = set_device(train_params["cuda_number"])
    print(f"Starting model validation stage on {data_path} set...")

    data_set = pd.read_csv(os.path.join(experiment_folder_path, data_path))
    data_loader = build_data_loader(data_set, Collection, train_params["batch_size"])

    if epoch_to_load is not None:
        model_path = os.path.join(experiment_folder_path, "model", f"best_model_{epoch_to_load}.pt")
        Model = load_model(model_path, Model, device)
    else:
        Model = prepare_model_for_eval(Model, device)

    path_to_save = os.path.join(experiment_folder_path, "results")
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    class_names = load_yaml(os.path.join(experiment_folder_path, "metadata.yaml"))["class_names"]
    df, result = _evaluate(Model, data_loader, data_set, device, class_names=class_names)

    # save everything
    if save_results:
        dataset_name = data_path.split("/")[-1].replace(".csv", "_results.csv")
        df.to_csv(os.path.join(path_to_save, dataset_name), index=False)
        result["data_path"] = data_path
        result["config"] = config
        # We use json, since there are problems with the float format in yaml when saving
        save_json(result, os.path.join(path_to_save, data_path.replace(".csv", "") + "_output.json"))
    print(f"Model evaluation is finished.")

    return df, result
