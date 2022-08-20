import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_proba_distr(pred: pd.DataFrame, map_target: dict, bins=30) -> None:
    """
    Plots probability distribution for correct and incorrect predictions for each class
    """
    for cls in map_target:
        correct = pred[(pred.true == map_target[cls]) & (pred.pred == map_target[cls])]
        incorrect = pred[(pred.true != map_target[cls]) & (pred.pred == map_target[cls])]
        plt.figure(
            figsize=(15, 5),
        )
        sns.set_style("darkgrid")
        plt.title(cls)
        plt.hist(correct[f"prob_{map_target[cls]}"], bins=bins, label="True", alpha=0.8)
        plt.hist(incorrect[f"prob_{map_target[cls]}"], bins=bins, label="False", alpha=0.4)
        plt.legend()
        plt.show()


def show_mistakes_by_target(
    pred: pd.DataFrame, target: int, map_target: dict,
    images_num: int = None, txt_color: str = 'w', figsize: int = 10
) -> None:
    """Shows images that are actually "target", but was predicted as smth else"""
    incorrect = pred[(pred.pred != target) & (pred.true == target)]
    trgt_clsname_dct = {map_target[k]: k for k in map_target}

    if images_num is not None:
        if len(incorrect) > images_num:
            incorrect = incorrect.iloc[:images_num]

    for idx, row in incorrect.iterrows():
        pred_lbl = row.pred

        txt = (
            f'True: {trgt_clsname_dct[target]} ({row[f"prob_{target}"]:.2f}); \n'
            + f'False: {trgt_clsname_dct[pred_lbl]} ({row[f"prob_{pred_lbl}"]:.2f})'
        )

        plt.figure(figsize=(figsize, figsize))
        plt.imshow(plt.imread(row["img_path"]))
        plt.text(0, 50, txt, fontsize=18, color=txt_color)
        plt.axis("off")
        plt.show()


def show_mistakes_by_pred(
    pred: pd.DataFrame, target: int,
    map_target: dict, txt_color: str = 'w', figsize: int = 10
) -> None:
    """Shows images that were predicted as target, but actually is not"""
    incorrect = pred[(pred.pred == target) & (pred.true != target)]
    trgt_clsname_dct = {map_target[k]: k for k in map_target}

    for idx, row in incorrect.iterrows():
        pred_lbl = row.pred
        trgt_lbl = row.target

        txt = (
            f'True: {trgt_clsname_dct[trgt_lbl]} ({row[f"prob_{trgt_lbl}"]:.2f}); \n'
            + f'False: {trgt_clsname_dct[pred_lbl]} ({row[f"prob_{pred_lbl}"]:.2f})'
        )

        plt.figure(figsize=(figsize, figsize))
        plt.imshow(plt.imread(row["img_path"]))
        plt.text(0, 50, txt, fontsize=18, color=txt_color)
        plt.axis("off")
        plt.show()
