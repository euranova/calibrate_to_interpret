
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torch import autograd

from .base import Explainer
from ...utils.utils import min_max_tensor
import pdb
import configparser
import os 

config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(), "code_final/cifar100/conf.ini"))


class GradCamExplainer(Explainer):
    """Grad cam explaine, use features maps to explain single prediction"""

    def __init__(self, model, transform=None):
        super(GradCamExplainer).__init__()
        self.features = []
        self.model = model
        self.derivative = []
        self.transform = transform
        self.__name__ = "GradCam"
        self.derivative = []
        if config["SCALER"]["method"] == "dirichlet":
            self.dirichlet = True
        else:
            self.dirichlet = False

    def _modify_gradient(self):
        """
            Get features maps during the forward pass
            Get gradient of score w.r.t the features maps during the backward pass
        Args:
            None
        Returns:
            None
        """

        def _forward_grad_cam_hook(module, input, output):
            """
            Copy activations into features list
            Args:
                module ([nn.module]): torch layer
                input ([torch tensor]): layer's input
                output ([torch tensor]): layer's output
            """
            self.features = torch.squeeze(output.detach().clone())

        modules = list(self.model.modules())
        layers = [module for module in modules if not isinstance(module, nn.Sequential)]

        found = False
        counter = 1
        if self.selected_layer is not None:
            print(layers[self.selected_layer + 1])
            self.handle = layers[self.selected_layer + 1].register_forward_hook(
                _forward_grad_cam_hook
            )
            found = True
        else:
            while not found or counter < len(layers) - 1:
                print(layers[-counter])
                if isinstance(layers[-counter], (nn.Conv2d)):
                    found = True
                    self.handle = layers[-counter].register_forward_hook(
                        _forward_grad_cam_hook
                    )
                    counter += 1
                    self.selected_layer = len(layers) - counter
                    break
                else:
                    counter += 1
        if not found:
            raise NotImplementedError("Convolutionnal layer not found")

    def compute_saliency(self, grad):
        """
            Compute saliency map by computing the linear combination of the features
            maps weigthened by the absolute average of the
            derivative w.r.t to themselfs
        Args:
            image: tensor input image
        Returns:
            heatmap: tensor saliency map
        """
        gradients = torch.mean(grad.squeeze(), (1, 2))
        heatmap = torch.zeros(self.features.shape[1:]).cpu()
        for i, alpha in enumerate(gradients):
            heatmap += (
                alpha.detach().clone().cpu()
                * self.features[i, :, :].detach().clone().cpu()
            )
        heatmap = nn.functional.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=([self.size_x, self.size_y]),
            scale_factor=None,
            mode="bilinear",
            align_corners=True,
            recompute_scale_factor=None,
        )

        # heatmap = nn.ReLU()(heatmap)
        heatmap = min_max_tensor(heatmap).squeeze().numpy()
        return heatmap

    def __call__(
        self,
        image,
        plot=True,
        calibrator=None,
        multiclass=True,
        verbose=True,
        output_class=None,
        selected_layer=None,
        device="cpu",
    ):
        # Initializing the image as a trainable objects to get its gradient with respect to the output
        image = self._initialisation(
            calibrator,
            multiclass,
            image,
            verbose,
            output_class=output_class,
            device="cpu",
        )
        self.model.eval()
        self.selected_layer = selected_layer
        self._modify_gradient()
        image, _, _ = self.first_forward_pass(image)
        self.handle.remove()
        grad = self.compute_partial_derivatives()
        saliency = self.compute_saliency(grad)
        if plot:
            self.plot(saliency, overlay=True)
        return saliency
