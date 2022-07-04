import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.patches import Rectangle
import sys
import os

sns.set_theme(style="whitegrid")
hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]


def hatches_plot(ax, h):
    ax.add_patch(
        Rectangle((3, -0.25), 0.4, 1.3, fill=True, hatch=h, color="peru", alpha=0.3)
    )


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(("outward", 10))

        else:
            spine.set_color("none")
    if "left" in spines:
        ax.yaxis.set_ticks_position("left")
    else:
        ax.yaxis.set_ticks([])

    if "bottom" in spines:
        ax.xaxis.set_ticks_position("bottom")
    else:
        ax.xaxis.set_ticks([])


def plot_auc_curves(model, dataset, calibrator):
    if calibrator == "temperature" and model == "vgg16":
        methods = [
            "Sensitivity",
            "IntegratedGradient",
            "MeaningfullPerturbation",
            "LRP",
            "Rise",
        ]
    else:
        methods = [
            "Sensitivity",
            "IntegratedGradient",
            "MeaningfullPerturbation",
            "Rise",
        ]
    plt.figure(figsize=(15, 5))
    for counter, method in enumerate(methods):
        colors = ["blue", "orange", "red", "green", "purple"]
        director = os.getcwd()
        df = pd.read_csv(
            os.path.join(
                director,
                "code_final/results/results_"
                + model
                + "_"
                + dataset
                + "_"
                + method
                + "_"
                + calibrator
                + ".csv",
            )
        )

        auc_calibrated = np.mean(
            [
                df["auc_calib_calib"]
                .apply(
                    lambda x: np.fromstring(
                        x.replace("\n", "")
                        .replace("[", "")
                        .replace("]", "")
                        .replace("  ", " "),
                        sep=" ",
                    )
                )
                .mean()
                for col in df.columns
            ],
            axis=0,
        )
        auc_non_calibrated = np.mean(
            [
                df["auc_non_calib"]
                .apply(
                    lambda x: np.fromstring(
                        x.replace("\n", "")
                        .replace("[", "")
                        .replace("]", "")
                        .replace("  ", " "),
                        sep=" ",
                    )
                )
                .mean()
                for col in df.columns
            ],
            axis=0,
        )

        plt.plot(
            auc_calibrated,
            label=method + " calibrated",
            color=colors[counter],
            linestyle="--",
        )
        plt.plot(
            auc_non_calibrated,
            label=method + " non calibrated",
            color=colors[counter],
        )
    auc_random = np.mean(
        [
            df["auc_random_calib"]
            .apply(
                lambda x: np.fromstring(
                    x.replace("\n", "")
                    .replace("[", "")
                    .replace("]", "")
                    .replace("  ", " "),
                    sep=" ",
                )
            )
            .mean()
            for col in df.columns
        ],
        axis=0,
    )
    auc_random_non_calib = np.mean(
        [
            df["auc_random_non_calib"]
            .apply(
                lambda x: np.fromstring(
                    x.replace("\n", "")
                    .replace("[", "")
                    .replace("]", "")
                    .replace("  ", " "),
                    sep=" ",
                )
            )
            .mean()
            for col in df.columns
        ],
        axis=0,
    )
    plt.plot(auc_random, label="random calibrated", color="black", linestyle="--")
    plt.plot(
        auc_random_non_calib,
        label="random non calibrated",
        color="black",
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    model = sys.argv[1]
    dataset = sys.argv[2]
    calibrator = sys.argv[3]
    plot_auc_curves(model, dataset, calibrator)
