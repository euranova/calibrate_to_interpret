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


def plot_ssim(model, dataset, calibrator):
    frames = []
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

    for method in methods:
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

        ssim = df["ssim"]
        ssim_sensitivity_ind = [method] * len(ssim)
        model_ind = [model + " " + dataset] * len(ssim_sensitivity_ind)
        df_modify = pd.DataFrame(
            {"method": ssim_sensitivity_ind, "ssim": ssim, "model": model_ind}
        )
        frames.append(df_modify)

    df = pd.concat(frames)
    plt.figure(figsize=(20, 10))
    plt.legend([], [], frameon=False)
    ax = sns.boxplot(x="method", y="ssim", data=df, hue="model", showfliers=False)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)

    # Might need to loop through the list if there are multiple lines on the plot
    ax.lines[0].set_linestyle("--")
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_ylim([-0.25, 1.01])
    # plt.legend([], [], frameon=False)
    plt.title("SSIM distribution using " + calibrator + " scaling   ", fontsize=17)
    plt.show()


if __name__ == "__main__":
    model = sys.argv[1]
    dataset = sys.argv[2]
    calibrator = sys.argv[3]
    plot_ssim(model, dataset, calibrator)
