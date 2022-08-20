import cv2
import numpy as np
import pandas as pd
from typing import Tuple


def calibrate_probas(pred_df: pd.DataFrame, mass_w: list, prob_cols: list, c: float = 1.0) -> pd.DataFrame:
    """
    Re-weighting of probabilities, taking into account
    the weight for each class (mass_w) and the strength of the re-weighting (c)
    """
    n_classes = len(mass_w)
    df = pred_df.copy()

    df[prob_cols] = df[prob_cols] / (mass_w**c)
    df[prob_cols] = df[prob_cols].div(df[prob_cols].sum(axis=1), axis=0)
    mapping_idx = {f"prob_{i}": i for i in range(n_classes)}
    df["pred"] = df[prob_cols].idxmax(axis=1).map(mapping_idx)

    return df


def get_values(df: pd.DataFrame) -> Tuple[list, list, list, list]:
    """
    Calculation of average intensity values in RGB space
    """

    def load_image(image_path):
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    images = [load_image(i) for i in df["img_path"]]

    red = [np.mean(images[idx][:, :, 0]) for idx in range(len(images))]
    green = [np.mean(images[idx][:, :, 1]) for idx in range(len(images))]
    blue = [np.mean(images[idx][:, :, 2]) for idx in range(len(images))]
    values = [np.mean(images[idx]) for idx in range(len(images))]

    return red, green, blue, values
