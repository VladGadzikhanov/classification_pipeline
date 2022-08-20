import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


def plot_confusion_matrix(cm, class_names, normalize=True):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=0)[np.newaxis, :]
    df_cm = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    )
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(df_cm, annot=True)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    ax.set_ylim(len(class_names), 0)
    plt.show()


def show_obj(img_paths):
    images = [Image.open(path) for path in img_paths]
    image = np.hstack(images)
    plt.figure(figsize=(20, 70))
    plt.imshow(image)
    plt.grid()
    plt.axis("off")
    plt.show()
