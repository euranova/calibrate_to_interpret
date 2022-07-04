
from abc import ABC, abstractmethod
import torch
from torch._C import BoolType
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ...layers.gaussian_kernel import GaussianLayer
from ...utils.utils import use_gpu


class SubNet(nn.Module):
    """
    Submodel used to compute intermediate gradient
    Creates a sequential model after a list of layers
    """

    def __init__(self, layers):
        super(SubNet, self).__init__()
        self.layers = layers

    def forward(self, x):
        cloned_x = x.clone()
        for layer in self.layers:
            try:
                cloned_x = layer(cloned_x)
            except Exception as e:
                cloned_x = cloned_x.flatten()
                cloned_x = layer(cloned_x)
        return cloned_x


class Explainer(ABC):
    """Base class for saliency maps makers objects
    size_x int : input image size
    size_y int : input image size
    baseline str  : Name of baseline method
    model pytorch module :
    base_img pytorch tensor : 3D image which is use
                as reference for mehtods based on reference
    sigma float : standart deviation for gaussian kernel
    predicted int : class predicted during the first forward pass
    """

    def __init__(self):
        super(Explainer).__init__()
        self.size_x = 0
        self.size_y = 0
        self.base_img = None
        self.model = None
        self.baseline = ""
        self.sigma = 0
        self.predicted = ""

    def prepare_data(self, path_img="data/cat.jpg", size=224):
        """
            Adapt image into a format of 224 by 224 by 3
            2D image so input image dim should be 3
            If transform attribute is none then default preprocessing
            is the one for imageNet
        Args:
            path_img: str path to the imput image
        Returns:
            image: Pytorch tensor
        """
        image = Image.open(path_img)
        self.non_transformed_img = transforms.Resize(size)(image)
        if self.transform is None:
            transform_ = transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
                ]
            )
        else:
            transform_ = self.transform
        image = transform_(image)
        self.size_x = image.shape[1]
        self.size_y = image.shape[2]
        return image

    @use_gpu
    def _prediction(self, image):
        """
        Deal with adding last layer if needed
            and does the prediction
        Args:
            image ([Pytorch tensor]):4D (1,channels,size_x,size_y)

        Returns:
            ypred ([Pytorch tensor]): 1D (1,nb_cl asses)
        """
        print(image.get_device())
        print(self.model.get_device())
        self.model = self.model.to(self.device)
        if self.multiclass:
            self.last_layer = nn.Softmax(dim=1).to(self.device)
        else:
            self.last_layer = nn.Sigmoid().to(self.device)
        if self.verbose:
            print("   forward step ...")
        if self.dirichlet == True:
            if self.calibrator is not None:
                ypred = torch.log(nn.Softmax(dim=-1)(self.model(image.float())))
                S_ = torch.hstack((ypred, torch.ones((len(ypred), 1)).to(self.device)))
                ypred = torch.mm(
                    S_,
                    torch.FloatTensor(self.calibrator.weights.transpose()).to(
                        self.device
                    ),
                )
                ypred = self.last_layer(ypred)
            else:
                ypred = self.last_layer(self.model(image.float()))
        else:
            if self.calibrator is not None:
                ypred = self.last_layer(
                    self.model(image.float()) * self.calibrator._weights[0]
                )
            else:
                ypred = self.last_layer(self.model(image.float()))
        return ypred

    def first_forward_pass(self, image, image_grad=False):
        """
            First forward pass, choice of class
            Before any methods the class of maximum score is selected as class of interest
        Args:
            image: tensor input image
        Returns:
            image: tensor resized
            ypred: tensor model's prediction
            onehot: one hot encoded tensor with 1 at the class predicted by the model
        """
        self.model.eval()
        self.model.zero_grad()
        if image.dim() == 3:
            if image_grad:
                with torch.no_grad():
                    image = image.unsqueeze(0)
            else:
                image = image.unsqueeze(0)
        image.requires_grad = True
        ypred = self._prediction(image)
        self.predicted = int(torch.argmax(ypred).cpu().detach().numpy())
        if self.output_class is None:
            score = torch.max(ypred).cpu().detach().numpy()
            onehot = torch.zeros((1, ypred.size()[-1]), dtype=int)
            onehot[0, self.predicted] = 1
        else:
            score = ypred[0, self.output_class].detach().numpy()
            onehot = torch.zeros((1, ypred.size()[-1]), dtype=int)
            onehot[0, self.output_class] = 1
            self.predicted = self.output_class
        if self.verbose:
            print(
                "Class {} predicted with a score of {} ".format(self.predicted, score)
            )
        return image, ypred, onehot

    def plot(self, saliency, overlay=True):
        """
            Basic ploting function overlaying the saliency map and the input image
        Args:
            saliency np array :  saliency map
            overlay Bool: Overlay heatmap on input image
        """
        assert isinstance(
            saliency, np.ndarray
        ), """Wrong saliency type, should be a numpy array"""
        alpha_heatmap = 1
        if overlay:
            pil_image = self.non_transformed_img
            plt.imshow(pil_image, alpha=0.5)
            alpha_heatmap = 0.5
        plt.imshow(saliency, cmap=matplotlib.cm.jet, alpha=alpha_heatmap)
        plt.title(self.__name__)
        plt.show()

    def resize_to_input_size(self, saliency, method="Bilinear"):
        """Update saliency map to correspond to input image

        Args:
            saliency (pytorch tensor): saliency map
            image (pytorch tensor): input image

        Returns:
            heatmap: numpy array saliency map
        """
        heatmap = transforms.ToPILImage()(saliency)
        if method == "Bilinear":
            heatmap = np.array(
                heatmap.resize(
                    (self.size_y, self.size_x),
                    Image.BILINEAR,
                )
            )
        elif method == "Bicubic":
            heatmap = np.array(
                heatmap.resize(
                    (self.size_y, self.size_x),
                    Image.BICUBIC,
                )
            )
        else:
            raise NotImplementedError("Selected interpolation method does not exist")
        return heatmap

    def generate_baseline_image(self, image):
        """
            Generate reference image
        Args:
            image: tensor input image
        Returns:
            None
        """
        assert isinstance(
            self.baseline, str
        ), """Invalid baseline type,should be a string """
        assert torch.is_tensor(image), """Invalid image type, should be a tensor"""
        if self.baseline == "black":
            self.base_img = torch.zeros(image.shape).clone().detach()
        elif self.baseline == "noise":
            self.base_img = image + (self.sigma) * torch.randn(image.shape)
        elif self.baseline == "gaussian":
            kernel = GaussianLayer(perturbation_size=[image.shape[2], image.shape[3]])
            self.base_img = kernel(image, mask=torch.ones(image.shape))
        elif self.baseline == "white":
            self.base_img = torch.ones(image.shape).clone().detach()
        else:
            raise NotImplementedError(
                """ Wrong baseline value :
                Should be in (black,noise,gaussian,white) """
            )

    def _initialisation(
        self,
        calibrator,
        multiclass,
        image,
        verbose,
        output_class=None,
        layer=None,
        device="cpu",
    ):
        """[Process input]
           If image is as path then load the image and preprocess it
           using default transform function or the transform function given
           as parameter
        Args:
            calibrator ([netcal object]):  scaling object
            multiclass ([Bool]): Use sigmoid function or Softmax at the model outputs
            image ([str or tensor]): Input image or path to the image
            verbose ([Bool]) : Printing steps
            output_class ([int]) : class of interest, default to None
            layer ([int]) : position number of the layer of interest, default to None
        Returns:
            image [pytorch tensor]: 4D tensor for the input image
        """
        assert isinstance(multiclass, type(True)) and isinstance(
            verbose, type(True)
        ), """Wrong multiclass value,
                                 should be a bool"""
        assert (
            isinstance(layer, int) or layer == None
        ), """Wrong layer type,should be a int or None"""
        assert (
            isinstance(output_class, int) or layer == None
        ), """Wrong output_class type, should be a int"""
        assert isinstance(image, str) or torch.is_tensor(
            image
        ), """Wrong image type, should be a path to the image or a pytorch tensor"""
        self.device = device
        self.calibrator = calibrator
        self.multiclass = multiclass
        self.layer = layer
        self.verbose = verbose
        self.output_class = output_class
        if isinstance(image, str):
            image = self.prepare_data(image)
        else:
            assert (
                image.dim() == 3
            ), """Invalid input dimension, 
                  input dimension should be 3 (channel,size_x,size_y)"""
            self.non_transformed_img = np.transpose(
                image.squeeze().cpu().detach().clone().numpy(), (1, 2, 0)
            )
            self.size_x = image.shape[1]
            self.size_y = image.shape[2]
        return image

    @use_gpu
    def _prediction(self, image):
        """
        Deal with adding last layer if needed
            and does the prediction
        Args:
            image ([Pytorch tensor]):4D (1,channels,size_x,size_y)

        Returns:
            ypred ([Pytorch tensor]): 1D (1,nb_cl asses)
        """
        self.model = self.model.to(self.device)
        if self.multiclass:
            self.last_layer = nn.Softmax(dim=1)
        else:
            self.last_layer = nn.Sigmoid()
        if self.verbose:
            print("   forward step ...")
        if self.dirichlet == True:
            if self.calibrator is not None:
                ypred = torch.log(nn.Softmax(dim=-1)(self.model(image.float())))
                S_ = torch.hstack((ypred, torch.ones((len(ypred), 1)).to(self.device)))
                ypred = torch.mm(
                    S_,
                    torch.FloatTensor(self.calibrator.weights.transpose()).to(
                        self.device
                    ),
                )
                ypred = self.last_layer(ypred)
            else:
                ypred = self.last_layer(self.model(image.float()))
        else:
            if self.calibrator is not None:
                ypred = self.last_layer(
                    self.model(image.float()) * self.calibrator._weights[0]
                )
            else:
                ypred = self.last_layer(self.model(image.float()))
        return ypred

    @use_gpu
    def compute_saliency_step(self, image, mask, model):
        """
            One prediction step (usefull for gpu use)
        Args:
            image (pytorch tensor): input image
            mask (pytorch tensor): feature map binarized
            model (pytorch module ): pytorch model
        Returns:
            ypred (pytorch tensor): model's prediction
        """
        input_img = image * mask
        if self.multiclass:
            last_layer = nn.Softmax(dim=-1)
        else:
            last_layer = nn.Sigmoid()
        if self.calibrator is not None:
            if self.dirichlet == True:
                if self.calibrator is not None:
                    ypred = torch.log(nn.Softmax(dim=-1)(model(input_img.float())))
                    S_ = torch.hstack(
                        (ypred, torch.ones((len(ypred), 1)).to(self.device))
                    )
                    ypred = torch.mm(
                        S_,
                        torch.FloatTensor(self.calibrator.weights.transpose()).to(
                            self.device
                        ),
                    )
                    ypred = self.last_layer(ypred)
                else:
                    ypred = last_layer(
                        model(input_img.float()) * self.calibrator._weights[0]
                    )
            else:
                ypred = last_layer(
                    model(input_img.float()) * self.calibrator._weights[0]
                )
        else:
            ypred = model(input_img.float())
        return ypred

    def compute_one_layer_grad(self, input_layer, SubNet, new_layers):
        self.new_model = SubNet(new_layers)
        self.new_model.eval()
        print(new_layers)
        output = self.new_model(input_layer)
        onehot = torch.zeros((output.size()[-1]), dtype=int)
        onehot[self.predicted] = 1
        output.backward(onehot)
        grad = input_layer.grad
        return grad

    def compute_partial_derivatives(self, layers=None):
        """Compute gradient of a layer w.r.t the output
           If input contains multiples layers, function
           returns a list of gradients

        Args:
            layers (list of int, optional): Number of the layer (same order as when print model). Defaults to None.

        Returns:
             grad (pythorch tensor or list of pytorch tensor): Returns gradient of layer(s) w.r.t output
        """

        if layers is None:
            modules = list(self.model.modules())
            layers = [
                module for module in modules if not isinstance(module, nn.Sequential)
            ]
            new_layers = list(
                iter([layers[i] for i in range(self.selected_layer + 2, len(layers))])
            )
            input_layer = self.features.unsqueeze(0)
            input_layer.requires_grad = True
            grad = self.compute_one_layer_grad(input_layer, SubNet, new_layers)
        else:
            grad = []
            for counter, layer_id in enumerate(layers):
                modules = list(self.model.modules())
                layers = [
                    module
                    for module in modules
                    if not isinstance(module, nn.Sequential)
                ]
                new_layers = list(
                    iter([layers[i] for i in range(layer_id + 1, len(layers))])
                )
                input_layer = self.features[counter].detach().clone().unsqueeze(0)
                input_layer.requires_grad = True
                local_gradient = self.compute_one_layer_grad(
                    input_layer, SubNet, new_layers
                )
                grad.append(local_gradient)
        return grad
