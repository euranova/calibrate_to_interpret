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
from .Dataloaders.dataset_preprocessor_cifar100 import DatasetPreprocessorCifar100
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


i = 0
acc = 0
counter = 0
torch.seed = 0
np.random.seed(0)
random.seed(0)

config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(), "code_final/cifar100/conf.ini"))


DEVICE = config["GENERAL"]["device"]


transform = transforms.Compose(
    [
        # resize it to the size indicated by `image_size`
        # transforms.Resize((224, 224)),
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


def confidences_from_scores(scores_matrix, predictions, model):
    res = np.max(scores_matrix, axis=1)
    return res


def load_data():
    preprocessor = DatasetPreprocessorCifar100()

    data = preprocessor.download_dataset_cifar(
        root=os.path.join(os.getcwd(), "code_final/cifar100/src/dataset/cifar100")
    )
    calib_size = 0.3
    calibration_set = int(len(data) * calib_size)
    test_set = int(len(data) * (1 - calib_size))
    return calibration_set, test_set, data


def load_model():
    if config["MODELS_ARCHITECTURE"]["model"] == "vgg16":
        model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True
        )
    elif config["MODELS_ARCHITECTURE"]["model"] == "resnet32":
        model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar100_resnet32", pretrained=True
        )
    else:
        raise ValueError("wrong model selected")
    model.eval()  # for evaluation
    return model


def compute_accuracy(model, data, plot_matix=False):
    print("Computing  accuracy ...  \n")
    acc = 0
    count, i = 0, 0
    calibration_set = 3000
    model.to(DEVICE)
    score_matrix = np.zeros((100, 100))
    length = len(list(iter(data)))
    for x, y in iter(data):
        print_trainning_steps(count, length)
        x = transform(x).to(DEVICE).unsqueeze(0)
        if i < calibration_set:
            i += 1
            count += 1
        else:
            prediction = nn.Softmax(dim=-1)(model(x)).cpu()
            if prediction.argmax() == y:
                acc += 1
            count += 1
            i += 1
            score_matrix[prediction.argmax(), y] += 1
    if plot_matix:
        plt.imshow(score_matrix)
        plt.show()
    print("\n accuracy is : {} ".format(acc / count))
    return acc / count


def calibrate_model(model, data, DEVICE, saved=False):
    print("Calibrating model ...  \n")
    model = model.to(DEVICE)
    wrapper = WrapperForCalibration(model)
    SCORE = np.array([])
    PRED = np.array([])
    Y = np.array([])
    i = 0
    length = 3000
    if saved:
        directory = os.getcwd()
        name = os.path.join(
            directory,
            "code_final/cifar100/{}_{}_{}_{}.obj".format(
                "calibrator",
                "cifar100",
                config["MODELS_ARCHITECTURE"]["model"],
                config["SCALER"]["method"],
            ),
        )
        try:
            with open(name, "rb") as f:
                temperature = pickle.load(f)
        except Exception as e:
            raise ValueError("No scaler object")
    else:
        for x, y in iter(data):
            print_trainning_steps(i, length)
            x = transform(x).unsqueeze(0)

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
            if i > calibration_set - 1:
                print("\n")
                break
        scores = np.array(SCORE)
        predictions = PRED
        if config["SCALER"]["method"] == "dirichlet":
            reg = 1e-2
            # Full Dirichlet
            temperature = FullDirichletCalibrator(reg_lambda=reg, reg_mu=None)
            temperature.fit(scores, Y)
        else:
            temperature = TemperatureScaling()
            temperature.fit(scores, Y)
    return wrapper, temperature


def compute_ece(wrapper, model, data, temperature):
    print("Computing expected calibration error ... \n")
    SCORE = np.array([])
    PRED = np.array([])
    Y = np.array([])
    i = 0
    X = []
    length = len(list(iter(data)))
    for x, y in iter(data):
        print_trainning_steps(i, length)
        if i < calibration_set:
            i += 1
        else:
            x = transform(x).unsqueeze(0)
            x = x.to(DEVICE)
            y = np.array(y)
            X.append(x)
            predictions, scores = wrapper.predict_both(x)
            prediction_calibrated, scores_calibrated = wrapper.predict_both(
                x, temperature=temperature, calib=True
            )
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
    if config["SCALER"]["method"] == "dirichlet":
        calibrated = temperature.predict_proba(scores)
    else:
        calibrated = temperature.transform(scores)

    calibrator = config["SCALER"]["method"]
    name_model = config["MODELS_ARCHITECTURE"]["model"]
    name = "code_final/cifar100/{}_{}_{}_{}.obj".format(
        "calibrator", "cifar100", name_model, calibrator
    )

    with open(name, "wb") as output:
        pickle.dump(temperature, output)

    conf = confidences_from_scores(scores, Y, model)
    conf_calibrated = confidences_from_scores(calibrated, Y, model)

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
    model, data, method, interpretation_method, transform, WDR, plot=False
):
    print("Deletion experiments ...  \n")
    directory = os.getcwd()
    print(config.keys)
    calibrator = config["SCALER"]["method"]
    name_model = config["MODELS_ARCHITECTURE"]["model"]
    name = os.path.join(
        directory,
        "code_final/cifar100/{}_{}_{}_{}.obj".format(
            "calibrator", "cifar100", name_model, calibrator
        ),
    )
    try:
        with open(name, "rb") as f:
            temperature = pickle.load(f)
    except Exception as e:
        raise ValueError("No scaler object")
    print(temperature)
    for dir in [
        "auc_vectors_calibrated",
        "auc_vectors_non_calibrated",
        "auc_vectors_random",
        "img_non_calib",
        "img_calib",
    ]:
        try:
            os.makedirs(
                "code_final/cifar100/{}/{}/{}/{}/{}".format(
                    WDR,
                    config["MODELS_ARCHITECTURE"]["model"],
                    method,
                    config["GENERAL"]["method"],
                    dir,
                )
            )
        except FileExistsError:
            # directory already exists
            pass

    method = methods_list[interpretation_method]
    counter = 0
    exp = method(model)
    model = model.to(DEVICE)
    for x, y in iter(data):
        x = transform(x)
        image = x.to(DEVICE)
        # prediction = model(image.unsqueeze(0)).cpu().detach().argmax()
        saliency_ref = exp(image, plot=plot, calibrator=None, device=DEVICE)
        saliency_calib = exp(image, plot=plot, calibrator=temperature, device=DEVICE)
        saliency_random = min_max_tensor(np.random.random(224 * 224))
        with open(
            r"code_final/cifar100/{}/{}/{}/{}/img_non_calib/img_non_calib_{}.npy".format(
                WDR, name_model, calibrator, interpretation_method, counter
            ),
            "wb",
        ) as f:
            np.save(f, saliency_ref)
        with open(
            r"code_final/cifar100/{}/{}/{}/{}/img_calib/img_calib_{}.npy".format(
                WDR, name_model, calibrator, interpretation_method, counter
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
            file=config["GENERAL"]["method"],
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
            file=config["GENERAL"]["method"],
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
            file=config["GENERAL"]["method"],
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
            file=config["GENERAL"]["method"],
            seed=0,
        )
        print(counter)
        counter += 1


if __name__ == "__main__":
    calibrator_name = config["SCALER"]["method"]
    interpretation_method = config["GENERAL"]["method"]
    name_model = config["MODELS_ARCHITECTURE"]["model"]
    name = "code_final/cifar100/{}_{}_{}_{}.obj".format(
        "calibrator", "cifar100", name_model, calibrator_name
    )

    model = load_model()
    calibration_set, test_set, data = load_data()

    # accuracy = compute_accuracy(model, data, plot_matix=True)

    wrapper, calibrator = calibrate_model(
        model, data, DEVICE=config["GENERAL"]["device"]
    )

    compute_ece(wrapper, model, data, calibrator)
    deletion_step(
        model,
        data,
        calibrator_name,
        interpretation_method,
        transform,
        "results",
        plot=False,
    )
