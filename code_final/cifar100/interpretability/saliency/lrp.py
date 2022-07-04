from urllib.request import HTTPHandler
import torch
import numpy as np
from torch.autograd.variable import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib

from .base import Explainer
from ...utils.utils import min_max_tensor
import configparser
import os

config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(), "code_final/cifar100/conf.ini"))


class LRPExplainer(Explainer):
    # pylint: disable=too-many-instance-attributes
    """LayerWise Relevance propagation
    rule str : retropropagation rule
    _jiter float : epsilon value of LRP-Eps
    gradient list : list of gradient
    layers list : list of layers
    handles list : list of hooks (used for clean model at the end)
    transform torchvision transform : preprocessor
    features list : list of activations maps
    """

    def __init__(self, model, rule="epsilon", epsilon=3, transform=None):
        super(LRPExplainer).__init__()
        self.model = model
        self.model.eval()
        self.features = []
        self._jitter = epsilon
        self.rule = rule
        self.gradients = []
        self.layers = []
        self.handles = []
        self.transform = transform
        self.__name__ = "LRP"
        if config["SCALER"]["method"] == "dirichlet":
            self.dirichlet = True
        else:
            self.dirichlet = False

    def _modify_gradient(self):
        """
            Get features maps of all layers during the forward pass
        Args:
            None
        Returns:
            None
        """

        def _forward_lrp_hook(module, input_layer, output_layer):
            # pylint: disable=unused-argument
            self.features.append(torch.squeeze(output_layer))

        self.layers = [
            module
            for module in self.model.modules()
            if not isinstance(module, nn.Sequential)
        ]
        counter = 0
        while counter < len(self.layers):
            self.handles.append(
                self.layers[counter].register_forward_hook(_forward_lrp_hook)
            )
            counter += 1

    def _relevance_conv(self, input_layer, layer, relevance):
        """
            Compute relevance to pass from a certain layer  to an upper layer
        Args:
            input_layer: tensor input of the layer
            layer: torch (nn.Functional) Upper layer
            relevance: float Relevance from lower layer
        """
        in_shpae = input_layer.shape
        print(in_shpae)
        try:
            z_score = self._jitter + layer.forward(input=input_layer)
        except Exception:
            old_inp_layer = input_layer
            try:  # if there a flatten layer
                input_layer = torch.flatten(input_layer)
                input_layer.retain_grad()
                z_score = self._jitter + layer.forward(input=input_layer)
            except Exception:  # if there expend of dimensions
                input_layer = old_inp_layer.unsqueeze(0)
                input_layer.retain_grad()
                z_score = self._jitter + layer.forward(input=input_layer)
        superior = (relevance / z_score).data
        (z_score * superior.data).sum().backward()
        conservation = input_layer.grad
        relevance = input_layer * conservation
        return relevance.view(in_shpae)

    def _backpropagate(self, image):
        """
            Backpropagate relevance through model
        Args:
            image: pytorch tensor input image
        Results:
            heatmap: final heatmap
        """
        _, ypred, onehot = self.first_forward_pass(image)
        ypred.backward(onehot)
        layers = [
            module
            for module in self.model.modules()
            if not isinstance(module, nn.Sequential)
        ]
        relevance_k = [None] * (len(layers) - 1) + [ypred]
        for counter, layer in enumerate(reversed(self.layers)):
            if counter > len(layers) - 3:
                break
            if isinstance(
                layer, (nn.Conv2d, nn.AvgPool2d, nn.Linear, nn.AdaptiveAvgPool2d)
            ):
                relevance_k[-(counter + 2)] = self._relevance_conv(
                    Variable(self.features[-(counter + 3)], requires_grad=True),
                    layer,
                    relevance_k[-(counter + 1)],
                )
            elif isinstance(layer, nn.MaxPool2d):
                layer = nn.AvgPool2d(2)
                relevance_k[-(counter + 2)] = self._relevance_conv(
                    Variable(self.features[-(counter + 3)], requires_grad=True),
                    layer,
                    relevance_k[-(counter + 1)],
                )
            else:
                relevance_k[-(counter + 2)] = relevance_k[-(counter + 1)]
        heatmap = relevance_k[1].sum(0).detach().numpy()
        heatmap = np.where(heatmap is np.NaN, 0, heatmap)
        # print(np.max(heatmap))
        heatmap = np.nan_to_num(heatmap)
        heatmap = min_max_tensor(heatmap)
        return heatmap

    def __call__(
        self,
        image,
        plot=True,
        calibrator=None,
        multiclass=True,
        verbose=True,
        output_class=None,
    ):
        """
            Apply the method
        Args:
            image: tensor input image
            plot: bool plot or not the saliency map once computed
            return_map: bool return saliency map as np.ndarray
        Returns:
            saliency: np.ndarray saliency map, only if return_map=True
        """
        try:
            image = self._initialisation(
                calibrator, multiclass, image, verbose, output_class=output_class
            )
            self._modify_gradient()
            saliency = self._backpropagate(image)
            if plot:
                self.plot(saliency, overlay=False)
            for handler in self.handles:
                handler.remove()
            return saliency
        except RuntimeError as error:
            raise NotImplementedError(
                """This method is not compatible with models having skip connections! 
                Try an other method or an other model"""
            ) from error
