
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from .base import Explainer
from ...utils.utils import print_trainning_steps, use_gpu
from ...layers.gaussian_kernel import GaussianLayer
import configparser
config = configparser.ConfigParser()
config.read("code_final/food101/conf.ini")

class MeaningfullPerturbationExplainer(Explainer):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals

    # All instances are used
    """Explain single model prediction by masking input image
    and try to solve optimisation problem"""

    def __init__(self, model, percentage=0.2, sigma=10, transform=None):
        super(MeaningfullPerturbationExplainer).__init__()
        self.model = model
        self.percentage = percentage
        self.sigma = sigma
        self.mask = []
        self.l1_norm_coeff = 0.1
        self.tv_coeff = 1
        self.learning_rate = 0.1
        self.upsampled_mask = []
        self.kernel = []
        self.blured_img = []
        self.real_img = []
        self.transform = transform
        self.__name__ = "MeaningfullPerturbation"
        if config["SCALER"]["method"] == "dirichlet":
            self.dirichlet = True
        else:
            self.dirichlet = False

    def _generate_mask(self):
        self.mask = Variable(torch.ones(28, 28).float())
        # self.mask[5:15,5:15] = torch.zeros((self.mask[5:15,5:15].shape))

    def _total_variation(self, image, tv_beta=3):
        img = image[0, 0, :]
        row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
        col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
        return row_grad + col_grad

    @use_gpu
    def training_step(
        self,
        position_max,
        optimizer,
        real_img,
        blured_img,
        upsampled_mask,
        model,
    ):
        # pylint: disable=too-many-arguments
        """
            Single step of gradient descent, compute model prediction on masked image and update
            mask according to gradient values + past score value (avoid adeversarial effects)
        Args:
            position_max: int number class of interest
            optimizer: pytorch optimizer
            real_img: pytorch tensor input image
            blured_img: pytorch tensor fully blured image
            upsampled_mask: pytorch tensor 3D mask at input image size
            model: pytorch module , model
        Returns:
            optimizer: pytorch optimizer
            self.mask: masked updated
        """
        masked_img = real_img * upsampled_mask + blured_img * (1 - upsampled_mask)
        # masked_img = masked_img + (0.1)*torch.randn(masked_img.shape).cuda()
        if self.calibrator is not None:
            if self.dirichlet == True:
                if self.calibrator is not None:
                    ypred = torch.log(nn.Softmax(dim=-1)(model(masked_img.float())))
                    S_ = torch.hstack(
                        (ypred, torch.ones((len(ypred), 1)).to(self.device))
                    )
                    ypred = torch.mm(
                        S_,
                        torch.FloatTensor(self.calibrator.weights.transpose()).to(
                            self.device
                        ),
                    )
                    score_masked_result = nn.Softmax(dim=-1)(ypred).cpu()
                else:
                    score_masked_result = nn.Softmax(dim=-1)(
                        model(masked_img.float()) * self.calibrator._weights[0]
                    ).cpu()
            else:
                score_masked_result = nn.Softmax(dim=-1)(
                    model(masked_img.float()) * self.calibrator._weights[0]
                ).cpu()
        else:
            score_masked_result = model(masked_img.float()).cpu()
        loss = (
            score_masked_result[0, position_max]
            + self.l1_norm_coeff * torch.mean(torch.abs(1 - self.mask))
            + self.tv_coeff * self._total_variation(self.mask)
        )
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.mask.data.clamp_(0, 1)
        return optimizer, self.mask

    def compute_saliency(self, image, ypred):
        # pylint: disable=too-many-function-args
        """
            Solve occlusion based optimization by gradient descent, try to minimize
            model's score modification due to mask and due to constrain
            of mask norm and total variation norm.
        Args:
            image: tensor input image
            ypred; tensor, tensor returned by the model with the full image
        Returns:
            masked_img: PIL image , masked input image
        """
        self.real_img = Variable(image, requires_grad=False)
        position_max = ypred.argmax().cpu().detach().numpy()
        self.upsampled_mask = self.mask.view(
            (1, 1, self.mask.shape[0], self.mask.shape[1])
        )
        self.mask = self.mask.view((-1, 1, self.mask.shape[0], self.mask.shape[1]))
        self.kernel = GaussianLayer(sigma=2)
        optimizer = optim.Adam([self.mask], lr=self.learning_rate)
        self.model.eval()
        self.mask.requires_grad_()
        self.mask.retain_grad()
        self.blured_img = self.kernel(
            image.clone().detach(), mask=torch.ones(image.shape), diff=False
        )
        self.blured_img = Variable(self.blured_img, requires_grad=False)
        for i in range(self.nb_steps):
            torch.autograd.set_detect_anomaly(True)
            self.upsampled_mask = nn.UpsamplingBilinear2d(
                size=(image.shape[-2], image.shape[-1])
            )(self.mask)
            self.upsampled_mask = self.upsampled_mask.tile(1, image.shape[1], 1, 1)
            print_trainning_steps(i, self.nb_steps)
            optimizer, self.mask = self.training_step(
                position_max,
                optimizer,
                self.real_img,
                self.blured_img,
                self.upsampled_mask,
                self.model,
            )
            optimizer.step()
        self.mask = nn.UpsamplingBilinear2d(size=(image.shape[-2], image.shape[-1]))(
            self.mask
        )
        self.mask = 1 - self.mask
        return self.mask.squeeze().detach().numpy()

    def plot(self, saliency, image=None):
        """
            Basic ploting function overlaying the saliency map and the input image
        Args:
            saliency: tensor saliency map
        Returns:
            None
        """
        pil_image = transforms.ToPILImage()(image.squeeze().detach()).convert("RGB")
        plt.imshow(pil_image, alpha=0.5)
        plt.imshow(saliency, alpha=0.5, cmap=matplotlib.cm.jet)
        plt.show()

    def __call__(
        self,
        image,
        plot=True,
        calibrator=None,
        multiclass=True,
        verbose=True,
        output_class=None,
        nb_steps=600,
        device="cpu",
    ):
        """
        Apply the method

        Args:
            image 'pytorch tensor or str): input image or path to image
            plot (bool, optional): plot the saliency map . Defaults to True.
            calibrator (netcal object, optional):  Temperature scaler. Defaults to None.
            multiclass (bool, optional): if yes then Softmax layer added. If False Sigmoid added. Defaults to True.
            verbose (bool, optional): Print steps during processing. Defaults to True.
            output_class (int, optional): Class of interest. If None argmax selected. Defaults to None.

        Returns:
            saliency (numpy array): saliency map
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
        self.nb_steps = nb_steps
        image, ypred, _ = self.first_forward_pass(image)
        self._generate_mask()
        saliency = self.compute_saliency(image, ypred)
        if plot:
            self.plot(saliency, image)
        return np.array(saliency)
