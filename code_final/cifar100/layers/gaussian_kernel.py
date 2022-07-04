import torch
import torch.nn as nn
import numpy as np
import scipy
import scipy.ndimage
from ..utils.utils import use_gpu


class GaussianLayer(nn.Module):
    """
    Gaussian kernel perturabtion

    """

    def __init__(self, perturbation_size=[11, 11], sigma=200):
        """
            Initialisation
        Args:
            perturbation_size: list of int   neighborhood size in two dimensions (x,y)
            sigma: int sigma parameter of the gaussian law
        Returns:
            None
        """
        super(GaussianLayer, self).__init__()
        self.perturbation_size = perturbation_size
        self.edge = [int(param / 2) for param in self.perturbation_size]
        self.sigma = sigma
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(
                (self.edge[0], self.edge[0], self.edge[1], self.edge[1])
            ),
            nn.Conv2d(
                3, 3, perturbation_size, stride=1, padding=0, bias=None, groups=3
            ),
        )
        self._weights_init()

    @use_gpu
    def forward(self, image, position=[0, 0], mask=None, diff=True):
        """
            Apply the perturbation according the given mask if so,
            if else perturb the neighborhood around the position given
            if mask is 2D , then the mask is tilled to dim (1,number of channels,size_x,size_y)
        Args:
            image: tensor input image
            position: list  list of len 2 containing x adnd y position in the center
                            of the region to perturb. Only used if mask is None
            mask: tensor binary mask with 1 for the region to perturb
            diff: Bool internal variable for some specific methods (difference analysis)
        Returns:
            image: pytorch image
        """
        if diff:
            blured_img = self.seq(image.unsqueeze(0)).squeeze()
        else:
            blured_img = self.seq(image)
        if mask is not None:
            if not isinstance(mask, torch.FloatTensor):
                mask = torch.from_numpy(mask)
            if mask.shape != image.shape:
                mask = mask.tile([1, image.shape[0], 1, 1])
            with torch.no_grad():
                mask_from_mask = torch.where(mask == 1, 1, 0).view(image.shape)
                image[mask_from_mask == 1] = blured_img[mask_from_mask == 1]
        else:
            left_edge = max(0, position[0] - self.edge[0])
            rigth_edge = min(position[0] + self.edge[0], image.shape[2])
            upper_egde = min(position[1] + self.edge[1], image.shape[3])
            lower_edge = max(position[1] - self.edge[1], 0)
            image[:, :, left_edge:rigth_edge, lower_edge:upper_egde] = blured_img[
                :, :, left_edge:rigth_edge, lower_edge:upper_egde
            ]
        return image

    @use_gpu
    def _weights_init(self):
        """
            Initialise the weights using scipy's gaussian law
        Args:
            None
        Returns:
            None
        """
        grid = np.zeros((self.perturbation_size[0], self.perturbation_size[1]))
        grid[self.edge[0], self.edge[1]] = 1
        k = scipy.ndimage.gaussian_filter(grid, sigma=self.sigma)
        for _, parameters in self.named_parameters():
            parameters.data.copy_(torch.from_numpy(k))
