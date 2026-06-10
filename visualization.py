import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrix():

    cm = np.array([
        [443, 67, 31],
        [118, 137, 24],
        [98, 49, 118]
    ])

    labels = [
        "Negative",
        "Neutral",
        "Positive"
    ]

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        "results/confusion_matrix.png",
        dpi=300
    )

    plt.close()


if __name__ == "__main__":
    plot_confusion_matrix()
