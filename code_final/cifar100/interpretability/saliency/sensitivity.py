import torch

from .base import Explainer
from ...utils.utils import min_max_tensor
import configparser
import os

config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(), "code_final/cifar100/conf.ini"))


class SensitivityExplainer(Explainer):
    """Simple gradient of score w.r.t input image
    transform ([pytorch transformer]): Image preprocessor
    """

    def __init__(self, model, transform=None):
        super(SensitivityExplainer).__init__()
        self.model = model
        self.__name__ = "Sensitivity"
        self.transform = transform
        print(config["SCALER"]["method"])
        if config["SCALER"]["method"] == "dirichlet":
            self.dirichlet = True
        else:
            self.dirichlet = False

    def compute_saliency(self, image):
        """
        Basic derivative of score w.r.t the input image (argmax of score used)
        Args:
            image: Tensor input image
            model: Pytorch trained model
            plot: Bool plotting results or not
        Returns:
            salency: numpy array saliency map
        """
        # first forward pass to get scores
        image, ypred, onehot = self.first_forward_pass(image, image_grad=True)
        image.requires_grad = True
        # print if wanted
        if self.verbose:
            print("   backward step ...")
        # compute gradient
        ypred.backward(onehot.to(self.device))
        # keep only positive contribution
        saliency = torch.sum(image.grad.data.abs(), dim=1).cpu()
        # normalize and cast as numpy array
        saliency = min_max_tensor(saliency).squeeze().detach().numpy()
        return saliency

    def __call__(
        self,
        image,
        plot=True,
        calibrator=None,
        multiclass=True,
        verbose=True,
        output_class=None,
        device="cpu",
    ):
        """
        Apply the method
        Args:
            image 'pytorch tensor or str): input image or path to image
            plot (bool, optional): plot the saliency map . Defaults to True.
        Returns:
            saliency (numpy array): saieny map
        """
        # Initializing the image as a trainable objects to get its gradient with respect to the output
        image = self._initialisation(
            calibrator,
            multiclass,
            image,
            verbose,
            output_class=output_class,
            device=device,
        )
        # Computing saliency map thanks using gradient of output w.r.t input
        saliency = self.compute_saliency(image)
        # Plotting result if asked to
        if plot:
            self.plot(saliency)
        return saliency
