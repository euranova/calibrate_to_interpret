from posix import listdir
import sys
from torchvision import models
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from PIL import Image
from .interpretability.saliency.sensitivity import SensitivityExplainer
from .metrics.mesures_mean import deletion
from netcal.scaling import TemperatureScaling
from skimage.metrics import structural_similarity as ssim
from .metrics.mesures_mean import deletion, deletion_random
from .utils.utils import min_max_tensor
from torchvision import transforms
import random
import configparser
from .calibration_wrapper import WrapperForCalibration
from ..expli.src.explicalib.calibration.evaluation.metrics.confidence.confidence_ece_ac import (
    confidence_ece_ac,
)
from ..expli.src.explicalib.calibration.evaluation.plots.confidence.confidence_reliability_curve import (
    plot_confidence_reliability_curve,
)

from ..dirichlet_python.dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
from ..dirichlet_python.dirichletcal.calib.matrixscaling import MatrixScaling
from .interpretability.saliency.grad_cam import GradCamExplainer
from .interpretability.saliency.integrated_gradient import IntegratedGradientExplainer
from .interpretability.saliency.sensitivity import SensitivityExplainer
from .interpretability.saliency.rise import RiseExplainer
from .interpretability.saliency.lrp import LRPExplainer
from .interpretability.saliency.meaningfull_perturbation import (
    MeaningfullPerturbationExplainer,
)


from torch.utils.data import Dataset, DataLoader
from .dataloader import Dataloader_food


i = 0
acc = 0
counter = 0
torch.seed = 0
np.random.seed(0)

config = configparser.ConfigParser()
config.read("code_final/food101/conf.ini")
print(list(config.keys()))


DEVICE = config["GENERAL"]["device"]
transform = transforms.Compose(
    [
        # resize it to the size indicated by `image_size`
        transforms.Resize((224, 224)),
        # convert it to a tensor
        transforms.ToTensor(),
        # normalize it to the range [−1, 1]
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

methods_list = {
    "Sensitivity": SensitivityExplainer,
    "IntegratedGradient": IntegratedGradientExplainer,
    "Rise": RiseExplainer,
    "MeaningfullPerturbation": MeaningfullPerturbationExplainer,
    "LRP": LRPExplainer,
}
data = DataLoader(
    torch.utils.data.Subset(Dataloader_food("test"), np.arange(5000)),
    batch_size=1,
    shuffle=True,
)

calibration_set = 2500


def print_trainning_steps(count, train_length):
    """
        Print  training state, ==>..|
    Args:
        count: int, actual training step
        train_length: int max training step
    Returns:
        None
    """
    sys.stdout.write(
        "\r"
        + "=" * int(count / train_length * 50)
        + ">"
        + "." * int((train_length - count) / train_length * 50)
        + "|"
        + " * {} %".format(int(count / train_length * 100))
    )
    sys.stdout.flush()
    if count == train_length:
        sys.stdout.write("\n")
        sys.stdout.flush()


def confidences_from_scores(scores_matrix):
    """Get max line from matrix score

    Args:
        scores_matrix (numpy array): [description]

    Returns:
        res (numpy array): vectors of maxs
    """
    res = np.max(scores_matrix, axis=1)
    return res


def load_model(name="vgg16"):
    """Load model and pretrained weights

    Args:
        name (str, optional): model's name (vgg16 or resnet32). Defaults to "vgg16".
    Returns:
        model (pytorch model): return pretrained model in eval mode
    """
    if name == "vgg16":
        model = models.vgg16(pretrained=True)
        num_features = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1]
        features.extend([nn.Linear(num_features, 101)])
        model.classifier = nn.Sequential(*features)
        dir = os.getcwd()
        name = os.path.join(dir, "code_final/food101/VGG16.h5")
        model.load_state_dict(torch.load(name, map_location=torch.device(DEVICE)))
    elif name == "resnet50":
        dir = os.getcwd()
        path = os.path.join(dir, "code_final/food101/resnetmodel.pth")
        checkpoint = torch.load(
            path,
            map_location=DEVICE,
        )
        model = models.resnet50(pretrained=False)
        classifier = nn.Linear(2048, 101)
        model.fc = classifier
        model.load_state_dict(checkpoint["model_state"], strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def compute_accuracy(model, plot_matix=False):
    """Compute accuracy on test set and plot confusion matrix

    Args:
        model (pytorch model): Pretrained model
        plot_matix (bool, optional): Plot confusion matrix. Defaults to False.

    Returns:
        acc : Accuracy
    """
    acc = 0
    count, i = 0, 0
    model.to(DEVICE)
    score_matrix = np.zeros((101, 101))
    length = 5000
    for x, y in data:
        print_trainning_steps(i, length)
        if i < calibration_set:
            i += 1
        else:
            prediction = nn.Softmax(dim=-1)(model(x.to(DEVICE))).cpu()
            if prediction.argmax() == y:
                acc += 1
            count += 1
            i += 1
            score_matrix[prediction.argmax(), y] += 1
    if plot_matix:
        plt.imshow(score_matrix)
        plt.show()
    return acc / count


def calibrate_model(model, method, DEVICE):
    """Apply post hoc calibration

    Args:
        model (pytorch model): Pretrained model
        method (str): name of calibration method (temperature or dirichlet)
        DEVICE (str): name of device (cpu or cuda:0 or cuda:1 ...)

    Returns:
        wrapper (WrapperForCalibration object): Wrapper for pytorch models to mimic Sklearn api
        temperature (FullDirichletCalibrator object or TemperatureScaling object) : trained calibrator object
    """
    model = model.to(DEVICE)
    wrapper = WrapperForCalibration(model)
    SCORE = np.array([])
    PRED = np.array([])
    Y = np.array([])
    i = 0
    for x, y in data:
        x = x.to(DEVICE)
        # np_x = np.array(x.cpu()).astype("float")
        predictions, scores = wrapper.predict_both(x)
        if i > 0:
            SCORE = np.concatenate((SCORE, np.array(scores)), axis=0)
            PRED = np.concatenate((PRED, [predictions]), axis=0)
            Y = np.concatenate((Y, [y]), axis=0)
        else:
            SCORE = np.array(scores)
            PRED = np.array([predictions])
            Y = np.array([y])
        i += 1
        print(i)
        if i > calibration_set - 1:
            break

    scores = np.array(SCORE)
    predictions = PRED
    if method == "dirichlet":
        reg = 1e-2
        # Full Dirichlet
        temperature = FullDirichletCalibrator(reg_lambda=reg, reg_mu=None)
        temperature.fit(scores, Y)
    else:
        temperature = TemperatureScaling()
        temperature.fit(scores, Y)

    return wrapper, temperature


def compute_ece(wrapper, model, temperature, method):
    """Compute Expected Calibration error using continous binning

    Args:
        wrapper (WrapperForCalibration object): Model with it's Sklearn api wrapper
        model (pytorch model): Pretrained model, unused
        temperature (FullDirichletCalibrator object or TemperatureScaling object): pretrained calibrator
        method (str ): calibration method's name
    """
    SCORE = np.array([])
    PRED = np.array([])
    Y = np.array([])
    i = 0
    X = []
    for x, y in data:
        if i < calibration_set:
            i += 1
        else:
            x = x.to(DEVICE)
            y = np.array(y)
            X.append(x)
            predictions, scores = wrapper.predict_both(x)
            prediction_calibrated, scores_calibrated = wrapper.predict_both(
                x, temperature=temperature, calib=True
            )
            print(scores.max(), scores_calibrated.max())
            if i > calibration_set + 2:
                SCORE = np.concatenate((SCORE, np.array(scores)), axis=0)
                SCORE_CALIB = np.concatenate(
                    (SCORE_CALIB, np.array(scores_calibrated)), axis=0
                )
                PRED = np.concatenate((PRED, [predictions]), axis=0)
                PRED_CALIB = np.concatenate(
                    (PRED_CALIB, [prediction_calibrated]), axis=0
                )
                Y = np.concatenate((Y, [y]), axis=0)
            else:
                SCORE = np.array(scores)
                SCORE_CALIB = np.array(scores_calibrated)
                PRED = np.array([predictions])
                PRED_CALIB = np.array([prediction_calibrated])
                Y = np.array([y])
            i += 1

    predictions = PRED
    prediction_calibrated = PRED_CALIB
    calibrated = SCORE_CALIB
    scores = SCORE
    scores = np.array(SCORE)
    predictions = PRED
    if method == "dirichlet":
        calibrated = temperature.predict_proba(scores)
    else:
        calibrated = temperature.transform(scores)

    calibrator = config["SCALER"]["method"]
    name_model = config["MODELS_ARCHITECTURE"]["model"]
    name = "code_final/food101/{}_{}_{}_{}.obj".format(
        "calibrator", "food101", name_model, calibrator
    )

    with open(name, "wb") as output:
        pickle.dump(temperature, output)

    conf = confidences_from_scores(scores)
    conf_calibrated = confidences_from_scores(calibrated)

    ece = confidence_ece_ac(
        model=wrapper,
        X=X,
        confidence_scores=conf,
        predictions=predictions,
        Y=Y,
    )
    print("ece non calib", ece)

    ece_b = confidence_ece_ac(
        model=wrapper,
        X=X,
        confidence_scores=conf_calibrated,
        predictions=prediction_calibrated,
        Y=Y,
    )
    print("ece calib", ece_b)

    plot_confidence_reliability_curve(
        model=wrapper,
        X=X,
        confidences=conf,
        predictions=predictions,
        Y=Y,
        n_bins=10,
        bandwidth=0.01,
        kernel="cosine",
        font=None,
    )

    plot_confidence_reliability_curve(
        model=wrapper,
        X=X,
        confidences=conf_calibrated,
        predictions=prediction_calibrated,
        Y=Y,
        n_bins=10,
        bandwidth=0.01,
        kernel="cosine",
        font=None,
    )


def deletion_step(
    model, method, interpretation_method, calibration_set, transform, WDR
):
    """Apply deletion experiment

    Args:
        model (pytorch model): Pretrained model
        method (str): name inteepretation algorithme
        interpretation_method (str): calibration method's name
        calibration_set (int): size calibration set
        transform (torchvision tranforms): preprocessing for images
        WDR (str): default root directory where the results are saved

    Raises:
        ValueError: [description]
    """
    dir = os.getcwd()
    calibrator = config["SCALER"]["method"]
    name_model = config["MODELS_ARCHITECTURE"]["model"]
    name = os.path.join(
        dir,
        "code_final/food101/{}_{}_{}_{}.obj".format(
            "calibrator", "food101", name_model, calibrator
        ),
    )
    try:
        with open(name, "rb") as f:
            temperature = pickle.load(f)
    except Exception as e:
        raise ValueError("No scaler object")
    method = methods_list[interpretation_method]
    counter = 0
    exp = method(model)
    model = model.to(DEVICE)
    try:
        with open(name, "rb") as f:
            temperature = pickle.load(f)
    except Exception as e:
        raise ValueError("No scaler object")
    for dir in [
        "auc_vectors_calibrated",
        "auc_vectors_non_calibrated",
        "auc_vectors_random",
        "img_non_calib",
        "img_calib",
    ]:
        try:
            os.makedirs(
                "code_final/food101/{}/{}/{}/{}/{}".format(
                    WDR,
                    config["MODELS_ARCHITECTURE"]["model"],
                    config["SCALER"]["method"],
                    config["GENERAL"]["method"],
                    dir,
                )
            )
        except FileExistsError:
            # directory already exists
            pass
    for i in range(calibration_set, 5000):
        name = str(calibration_set + i)
        file = "img_food/img_" + name + ".npy"
        print(file)
        dir = os.getcwd()
        name = os.path.join(dir, file)
        try:
            img = np.load(file)
        except Exception as e:
            print(e)
        img = Image.fromarray(img)

        X = transform(img)
        image = X.to(DEVICE)
        # prediction = model(image.unsqueeze(0)).cpu().detach().argmax()
        saliency_ref = exp(image, plot=True, calibrator=None, device=DEVICE)
        saliency_calib = exp(image, plot=True, calibrator=temperature, device=DEVICE)
        saliency_random = min_max_tensor(np.random.random(224 * 224))
        with open(
            r"code_final/food101/{}/{}/{}/{}/img_non_calib/img_non_calib_{}.npy".format(
                WDR,
                name_model,
                config["SCALER"]["method"],
                exp.__name__.replace("Explainer", ""),
                counter,
            ),
            "wb",
        ) as f:
            np.save(f, saliency_ref)
        with open(
            r"code_final/food101/{}/{}/{}/{}/img_calib/img_calib_{}.npy".format(
                WDR,
                name_model,
                config["SCALER"]["method"],
                exp.__name__.replace("Explainer", ""),
                counter,
            ),
            "wb",
        ) as f:
            np.save(f, saliency_calib)
        auc_ref = deletion(
            model,
            image,
            saliency_ref,
            plot=False,
            auc_value=True,
            background="Blured",
            gpu=True,
            name=counter,
            calibrated="non_calibrated",
            file=exp.__name__.replace("Explainer", ""),
        )
        auc_calib = deletion(
            model,
            image,
            saliency_calib,
            plot=False,
            auc_value=True,
            background="Blured",
            gpu=True,
            name=counter,
            calibrated=temperature,
            file=exp.__name__.replace("Explainer", ""),
        )

        auc_random = deletion_random(
            model,
            image,
            saliency_random,
            plot=False,
            auc_value=True,
            background="Blured",
            gpu=False,
            name=counter,
            file=exp.__name__.replace("Explainer", ""),
        )

        auc_random_2 = deletion_random(
            model,
            image,
            saliency_random,
            plot=False,
            auc_value=False,
            background="Blured",
            gpu=False,
            name=counter,
            calibrated=temperature,
            file=exp.__name__.replace("Explainer", ""),
            seed=0,
        )
        print(counter)
        counter += 1


if __name__ == "__main__":
    calibrator_name = config["SCALER"]["method"]
    interpretation_method = config["GENERAL"]["method"]
    name_model = config["MODELS_ARCHITECTURE"]["model"]
    name = "code_final/food101/{}_{}_{}_{}.obj".format(
        "calibrator", "food101", name_model, calibrator_name
    )

    model = load_model(name_model)
    accuracy = compute_accuracy(model, plot_matix=True)
    print("{}'s accuracy is {} :", accuracy)
    wrapper, calibrator = calibrate_model(
        model, calibrator_name, DEVICE=config["GENERAL"]["device"]
    )

    compute_ece(wrapper, model, calibrator, calibrator_name)
    deletion_step(
        model, calibrator_name, interpretation_method, 2500, transform, "results_food"
    )
