
import warnings

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate

from .base import Explainer
from ...utils.utils import min_max_tensor, print_trainning_steps, use_gpu
import configparser
config = configparser.ConfigParser()
config.read("code_final/food101/conf.ini")

class RiseExplainer(Explainer):
    """ " Explaination by randomly masking the input image
    masks list : list of randomly sampled masks
    """

    def __init__(self, model, transform=None):
        """
            initialisation of the explainer that implement the method
            from http://bmvc2018.org/contents/papers/1064.pdf
        Args:
            model: pytorch model
        """
        super(RiseExplainer).__init__()
        self.model = model
        self.masks = []
        self.transform = transform
        self.__name__ = "Rise"
        if config["SCALER"]["method"] == "dirichlet":
            self.dirichlet = True
        else:
            self.dirichlet = False

    def _generate_random_masks(self, image, heigth=10):
        """
            Generate score according to paper's protocole
            Randomly generate small masks (size=(h,w) of [0,1])
            Resize it at input image size
            Binarize it again
        Args:
            image: tensor input image
            N: number of random mask generated
            h: small mask size
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            widght = heigth
            x_vec = np.linspace(0, 1, heigth)
            y_vec = np.linspace(0, 1, widght)
            x_grid, y_grid = np.meshgrid(x_vec, y_vec)
            shift_x = np.random.randint(0, image.shape[1])
            shift_y = np.random.randint(0, image.shape[2])
            mask = (np.random.binomial(size=(heigth, widght), n=1, p=0.6) > 0.5).astype(
                int
            )
            mask = np.logical_not(mask).astype(int)
            interolator = interpolate.interp2d(x_grid, y_grid, mask, kind="cubic")
            x_upscale_vec = np.linspace(0, 1, image.shape[2])
            y_upscale_vec = np.linspace(0, 1, image.shape[1])
            mask = interolator(x_upscale_vec, y_upscale_vec)
            mask[np.where(mask > 0.5)] = 1
            mask[np.where(mask < 0.5)] = 0
            mask = 1 - mask
            mask = np.roll(mask, shift_x, axis=0)
            mask = np.roll(mask, shift_y, axis=1)
            return mask

    @use_gpu
    def compute_saliency_step(
        self, squeezed_image, classe, mask, model, final_mask, count
    ):
        # pylint: disable=too-many-arguments
        """single step (prediction of one masked image)

        Args:
            squeezed_image pytorch tensor: 3D input image
            ypred pytorch tensor: tensor of model's output when the image is not masked
            classe ([type]): selected class
            mask pytorch tensor: random mask
            model ([type]): pytorch module
            final_mask pytorch tensor: mask weightened by the score obtained by the masked image

        Returns:
            [type]: [description]
        """
        input_img = squeezed_image * mask
        input_img = input_img.unsqueeze(dim=0)
        ypred = super(RiseExplainer, self).compute_saliency_step(
            input_img, torch.ones_like(input_img), model
        )
        intermediate_mask = ((ypred[0, classe]) / (count * self.n_smaples)) * mask
        final_mask += intermediate_mask
        return final_mask

    def compute_saliency(self, image, classe):
        """
            Executre model with randomly masked inputs, take the score and do a linear
            combinaison of these masks weigthened by their respectives scores
        Args:
            image: tensor input image
            ypred: tensor result returned by the model when the input is the full image
            classe: int class of interest
        Returns:
            final_mask: tensor saliency map
        """
        # kernel = GaussianLayer()
        final_mask = torch.zeros(image.shape[:])
        squeezed_image = torch.squeeze(image)
        with torch.no_grad():
            self.model.eval()
            for count in range(self.n_smaples):
                mask = self._generate_random_masks(image, heigth=8)
                norm = np.sum(mask)
                print_trainning_steps(count, self.n_smaples)
                mask = torch.from_numpy(mask).tile(squeezed_image.shape[0], 1, 1)
                final_mask = self.compute_saliency_step(
                    squeezed_image, classe, mask, self.model, final_mask, norm
                )
            final_mask = min_max_tensor(final_mask.cpu()).detach().numpy()
        return final_mask[0, :, :]

    def plot(self, saliency, image):
        """
            Basic ploting function overlaying the saliency map and the input image
        Args:
            saliency: tensor saliency map
            image: tensor input image
        Returns:
            None
        """
        saliency = saliency[0, :, :]
        heatmap = transforms.ToPILImage()(saliency.detach())
        pil_image = self.non_transformed_img
        heatmap = np.array(
            heatmap.resize(
                (self.size_y, self.size_x),
                Image.BILINEAR,
            )
        )
        plt.imshow(heatmap, cmap=matplotlib.cm.jet, alpha=0.7)
        plt.imshow(pil_image, alpha=0.3)
        plt.title("RISE")
        plt.show()

    def __call__(
        self,
        image,
        n_smaples=4000,
        plot=True,
        calibrator=None,
        multiclass=True,
        verbose=True,
        output_class=None,
        device="cpu",
    ):
        # pylint: disable=too-many-arguments
        """
            Apply the method
        Args:
            image: tensor input image
            N: number of random masks to use
            plot: bool plot or not the saliency map once computed
            return_map: bool return saliency map as np.ndarray
        Returns:
            saliency: np.ndarray saliency map, only if return_map=True
        """
        image = self._initialisation(
            calibrator,
            multiclass,
            image,
            verbose,
            output_class=output_class,
            device=device,
        )
        self.n_smaples = n_smaples
        _, ypred, _ = self.first_forward_pass(image)
        if self.output_class is None:
            self.output_class = ypred.argmax().detach().cpu().numpy()
        saliency = self.compute_saliency(image, self.output_class)
        if plot:
            self.plot(saliency, image)
        return np.array(saliency)
