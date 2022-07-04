import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import configparser
import os
import sys

config = configparser.ConfigParser()
config.read("code_final/results/conf.ini")
NAME_MODEL = config["MODELS_ARCHITECTURE"]["model"]


def total_variation(image, tv_pow=3):
    img = image
    row_grad = np.power(np.mean(np.abs((img[:-1, :] - img[1:, :]))), tv_pow)
    col_grad = np.power(np.mean(np.abs((img[:, :-1] - img[:, 1:]))), tv_pow)
    return row_grad + col_grad


def min_max_tensor(tensor):
    """
        Normalize tensor
    Args:
        tensor: tensor
    Returns:
        tensor: tensor normalized
    """
    if np.max(tensor) == np.min(tensor):
        tensor = np.zeros(tensor.shape)
    else:
        tensor -= np.min(tensor)
        tensor = tensor.astype("float")
        tensor /= np.max(tensor) - np.min(tensor)
    return tensor


model = config["MODELS_ARCHITECTURE"]["model"]
dataset = config["DATASET"]["data"]
method = config["GENERAL"]["method"]
calibration_name = config["SCALER"]["method"]


def compute_TV(model, dataset, calibration_name):
    if calibration_name == "temperature" and model == "vgg16":
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
        try:
            for method in methods:
                if dataset == "food101":
                    WDR = "results_food"
                elif dataset == "cifar100":
                    WDR = "results"
                else:
                    raise ValueError("Wrong dataset name in the config file")
                directory = os.path.join(
                    os.getcwd(),
                    "code_final/{}/{}/{}/{}/{}".format(
                        dataset, WDR, model, calibration_name, method
                    ),
                )
                path_calib = directory + "/img_calib/".format(method)
                path_non_calib = directory + "/img_non_calib/".format(method)
                mean_calib = []
                mean_non_calib = []
                mean_calib_sum = []
                mean_non_calib_sum = []

                lim = len(os.listdir(path_calib))
                for i in range(lim):
                    with open(path_calib + "img_calib_{}.npy".format(i), "rb") as file:
                        img_calib = np.load(file)
                    with open(
                        path_non_calib + "img_non_calib_{}.npy".format(i), "rb"
                    ) as file:
                        img_non_calib = np.load(file)
                    if not np.max(img_non_calib) < 1:
                        img_non_calib = min_max_tensor(img_non_calib)
                    if not np.max(img_calib) < 1:
                        img_non_calib = min_max_tensor(img_non_calib)

                    thresh = threshold_otsu(img_calib)
                    binary_mask_calib = img_calib > thresh

                    thresh = threshold_otsu(img_non_calib)
                    binary_mask_noncalib = img_non_calib > thresh

                    mean_calib.append(total_variation(binary_mask_calib.astype("int")))
                    mean_non_calib.append(
                        total_variation(binary_mask_noncalib.astype("int"))
                    )
                    mean_calib_sum.append(np.sum(binary_mask_calib.astype("int")))
                    mean_non_calib_sum.append(
                        np.sum(binary_mask_noncalib.astype("int"))
                    )
                print(
                    "{} mean tv after otzu binarisation after calibration : {} \n computed on {} values ".format(
                        method, np.mean(np.array(mean_calib)), i
                    )
                )
                print(
                    "{} mean tv after otzu binarisation before calibration : {} \n computed on {} values \n".format(
                        method, np.mean(np.array(mean_non_calib)), i
                    )
                )
        except Exception as e:
            print(
                "Missing data for : {}".format(method),
            )


if __name__ == "__main__":
    model = sys.argv[1]
    dataset = sys.argv[2]
    calibrator = sys.argv[3]
    compute_TV(model, dataset, calibrator)
