
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from .base import Explainer
from ...utils.utils import min_max_tensor, print_trainning_steps, use_gpu
import configparser
config = configparser.ConfigParser()
config.read("code_final/food101/conf.ini")

class IntegratedGradientExplainer(Explainer):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals

    # All instances are used
    """Integrated Gradient explains single prediction by averaging
    the derivative of the outputs w.r.t differents input images
    """

    def __init__(self, model, samples=30, sigma=0.3, transform=None):
        """
            initialisation of the explainer that implement the method
            from https://arxiv.org/pdf/1703.01365.pdf
        Args:
            model: pytorch model
            baseline: str select baseline image
            samples: int number of interpolated image between ref and input image
            sigma: float variance of gaussian law in smoothgrad
            transform : torchvision transformer image's preprocessor
        Returns:
            None
        """
        super(IntegratedGradientExplainer).__init__()
        self.model = model
        self.samples = samples
        self.sigma = sigma
        self.images = []  # save the interpolated images
        self.factors = []  # save associated values
        self.transform = transform
        self.__name__ = "IntegratedGradient"
        self.model.eval()
        if config["SCALER"]["method"] == "dirichlet":
            self.dirichlet = True
        else:
            self.dirichlet = False

    def _compute_riemann_integral(self, gradients):
        grad = gradients[:-1] + gradients[1:]
        grad = [gradient / 2 for gradient in gradients]
        return grad

    def _generate_convex_path(self, image):
        """
            Generate list of image along the convex path from the baseline image
            to the input image
        Inpouts :
            image: tensor input image
        Returns:
            None
        """

        if torch.cuda.is_available():
            image = image.to("cuda:0")
            self.base_img = self.base_img.to("cuda:0")
            clone = image.clone().detach().to("cuda:0")
        else:
            clone = image.clone().detach()
        self.images = [
            self.base_img + i / self.samples * (clone - self.base_img)
            for i in range(self.samples + 1)
        ]
        self.factors = [
            (1 / self.samples * (clone - self.base_img)).detach()
            for _ in range(self.samples)
        ]

    @use_gpu
    def compute_saliency(self, image, yreal):
        """
            Compute saliency map by averaging derivatives of scores w.r.t images
            from convex path
        Args:
            image: tensor input image
            yreal: pytorch tensor, output when model applied to input image
        Returns:
            saliency: tensor saliency map
        """
        saliency_list = []
        saliency = torch.zeros((image.shape))
        count = 0
        for input_image, factor in zip(self.images, self.factors):
            count += 1
            if input_image.dim() == 3:
                input_image = input_image.unsqueeze(0)
            self.model.zero_grad()
            input_image.requires_grad_()
            input_image.retain_grad()
            if self.verbose:
                print_trainning_steps(count, len(self.images))
            ypred = self.compute_saliency_step(
                input_image, torch.ones_like(input_image), self.model
            )
            onehot = torch.zeros(1, ypred.size()[-1], dtype=float)
            onehot[0, yreal.argmax().cpu().detach().numpy()] = 1
            onehot = onehot.to("cuda:0")
            ypred.backward(onehot)
            proxy = torch.squeeze(input_image.grad.data, dim=0).cpu()
            saliency_list.append(
                factor.cpu() * torch.from_numpy(proxy.detach().numpy())
            )
        saliency_list = self._compute_riemann_integral(saliency_list)
        for i in range(len(saliency_list)):
            saliency += saliency_list[i]
        saliency = saliency.squeeze().abs().sum(dim=0)
        # saliency = min_max_tensor(saliency)
        saliency = saliency.detach().numpy()
        return saliency

    def __call__(
        self,
        image,
        plot=True,
        use_smoothgrad=False,
        baselines=["white", "black"],
        n_smooth=10,
        sigma_smooth=0.3,
        calibrator=None,
        multiclass=True,
        verbose=False,
        output_class=None,
        device="cuda:0",
    ):
        """
            Apply the method
        Args:
            image: tensor input image
            plot: bool plot or not the saliency map once computed
            return_map: bool return saliency map as np.ndarray
            use_smoothgrad: bool apply smoothgrad method over integrated gradient
            N_smooth: int how many time smoothgrad is applied
            sigma_smooth: float variance of the gaussian law in smoothgrad
        Returns:
            saliency: np.ndarray saliency map, only if return_map=True
        """
        self.device = device
        # Initializing the image as a trainable objects to get its gradient with respect to the output
        assert isinstance(baselines, list), "baselines should be a list of strings"
        image = self._initialisation(
            calibrator,
            multiclass,
            image,
            verbose,
            output_class=output_class,
            device=device,
        )
        results = []
        final_saliency = torch.zeros(self.size_x, self.size_y)
        for baseline in baselines:
            self.baseline = baseline
            self.generate_baseline_image(image)
            self._generate_convex_path(image)
            image, ypred, _ = self.first_forward_pass(image)
            saliency = self.compute_saliency(image, ypred)
            if not use_smoothgrad:
                image = image.detach().clone()
                results.append(saliency)
            if use_smoothgrad:
                for _ in range(n_smooth - 1):
                    image = image + (sigma_smooth) * torch.randn(image.shape)
                    self.generate_baseline_image(image)
                    self._generate_convex_path(image)
                    saliency += self.compute_saliency(image, ypred)
                saliency = 1 / n_smooth * saliency
                results.append(saliency)
        for result in results:
            final_saliency += result / len(results)
        if plot:
            self.plot(saliency, overlay=False)
        return saliency
