import numpy as np
import torch
import torch.nn as nn
import configparser
import os

config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(), "code_final/cifar100/conf.ini"))
DEVICE = config["GENERAL"]["device"]
WDR = config["GENERAL"]["WDR"]


class WrapperForCalibration:
    """
    Sklearn interface
    """

    def __init__(self, model):
        self.model = model
        self.classes_ = np.arange(101)

    def predict_unique(self, images, proba=False):
        """
            Prediction of one image
            Check that the image is 4D (compatibility pytorch)
            Apply model
            append the results to list (if multiples images ) or return the prediction
        Args:
            image, pytorch tensor or list of pytorch tensors, input(s) image(s)
        Returns:
            predictions, pytorch tensor or list of pytorch tensors, output results
        """
        if proba == True:
            predictions = []
            if len(images) > 3:
                for image in images:
                    while image.dim() < 4:
                        image = torch.unsqueeze(image, 0)
                    predictions.append(self.model(image).cpu()).detach().numpy()
            else:
                while images.dim() < 4:
                    images = torch.unsqueeze(images, 0)
                predictions = self.model(images).cpu().detach().numpy()
        else:
            predictions = []
            if len(images) > 3:
                for image in images:
                    while image.dim() < 4:
                        image = torch.unsqueeze(image, 0)
                    predictions.append(self.model(image).cpu().detach().numpy())
            else:
                while images.dim() < 4:
                    images = torch.unsqueeze(images, 0)
                predictions = self.model(images).cpu().detach().numpy()
        return predictions

    def predict(self, images, numerical=False):
        """
            Sklearn behaviour , output the classes as defined by user (str or int)
        Args:
            image: pytorch tensor, input image
        Returns:
            predictions: np.array, list of predictions
        """
        predictions = []
        if len(images) > 3:
            for image in images:
                while image.dim() < 4:
                    image = torch.unsqueeze(image, 0)
                if numerical:
                    predictions.append(
                        self.classes_[
                            int(self.model(image).cpu().argmax().detach().numpy())
                        ]
                    )
                else:
                    predictions.append(
                        int(self.model(image).cpu().argmax().detach().numpy())
                    )
        else:
            while images.dim() < 4:
                images = torch.unsqueeze(images, 0)
            if numerical:
                predictions = self.classes_[
                    int(self.model(images).argmax().cpu().detach().numpy())
                ]
            else:
                predictions = int(self.model(images).argmax().cpu().detach().numpy())
        return np.array(predictions)

    def predict_both(
        self, images, numerical=False, dirichlet=False, temperature=None, calib=False
    ):
        """
            Sklearn behaviour , output the classes as defined by user (str or int)
        Args:
            image: pytorch tensor, input image
        Returns:
            predictions: np.array, list of predictions
        """
        predictions = []
        probas = []
        dirichlet = "dirichlet" == config["SCALER"]["method"]
        if len(images) > 3:
            for image in images:
                while image.dim() < 4:
                    image = torch.unsqueeze(image, 0)
                if not dirichlet and not (temperature is None):
                    prediction = self.model(image) * temperature._weights[0]
                elif dirichlet and not (temperature is None):
                    ypred = torch.log(nn.Softmax(dim=-1)(self.model(images)))
                    S_ = torch.hstack((ypred, torch.ones((len(ypred), 1)).to(DEVICE)))
                    prediction = torch.mm(
                        S_,
                        torch.FloatTensor(temperature.weights.transpose()).to(DEVICE),
                    )
                else:
                    prediction = self.model(image)
                if numerical:
                    predictions.append(
                        self.classes_[int(prediction.cpu().argmax().detach().numpy())]
                    )
                else:
                    predictions.append(int(prediction.cpu().argmax().detach().numpy()))
                probas.append(
                    nn.Softmax(dim=1)(prediction.cpu()).squeeze().detach().numpy()
                )
        else:
            while images.dim() < 4:
                images = torch.unsqueeze(images, 0)
            if not dirichlet and not temperature is None:
                prediction = self.model(images) * temperature._weights[0]
            elif not dirichlet and not temperature is None:
                ypred = torch.log(nn.Softmax(dim=-1)(self.model(images)))
                S_ = torch.hstack((ypred, torch.ones((len(ypred), 1)).to(DEVICE)))
                prediction = torch.mm(
                    S_, torch.FloatTensor(temperature.weights.transpose()).to(DEVICE)
                )
            else:
                prediction = self.model(images)
        if numerical:
            predictions = self.classes_[int(prediction.argmax().cpu().detach().numpy())]
        else:
            predictions = int(prediction.argmax().cpu().detach().numpy())
        probas.append(nn.Softmax(dim=-1)(prediction.cpu()).squeeze().detach().numpy())
        return np.array(predictions), np.array(probas)

    def predict_proba(self, images):
        """
            Sklearn behaviour , output vector of probabilities
        Args:
            image: pytorch tensor, input image
        Returns:
            predictions: np.array, list of probabilities (Softmax(scores))
        """
        predictions = []
        for image in images:
            prediction = self.predict_unique(image, proba=True)
            """while image.dim() < 4 :
                image = torch.unsqueeze(image, 0)"""
            predictions.append(
                nn.Softmax(dim=1)(torch.FloatTensor(prediction))
                .squeeze()
                .detach()
                .numpy()
            )
        return np.array(predictions)

    def _softmax(self, alphas):
        """
            Softmax of the input list in order to have sum to one
        Args:
            alphas: list list of weights
        Returns:
            alphas: list list of weights
        """
        alphas_exp = [np.exp(alpha) for alpha in alphas]
        sum_exp = sum(alphas_exp)
        alphas = alphas_exp / sum_exp
        return alphas

    def eval(self):
        return self.model.eval()

    def zero_grad(self):
        return self.model.zero_grad()

    def __call__(self, x):
        return self.model(x)
