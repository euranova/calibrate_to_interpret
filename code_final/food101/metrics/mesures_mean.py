import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn import metrics
from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage.segmentation import watershed, mark_boundaries, slic
from torchvision import transforms
import pickle
import configparser
from ..utils.utils import min_max_tensor, use_gpu
from ..layers.gaussian_kernel import GaussianLayer

np.random.seed(0)

config = configparser.ConfigParser()
config.read("code_final/food101/conf.ini")
print(list(config.keys()))
DEVICE = config["GENERAL"]["device"]
WDR = config["GENERAL"]["WDR"]
NAME_MODEL = config["MODELS_ARCHITECTURE"]["model"]


class RandomManager:
    def __init__(self, i):
        self.seed = i

    def __enter__(self):
        np.random.seed(self.seed)

    def __exit__(self, *args):
        np.random.seed(0)


def intersection_over_union(saliency, ground_truth_mask, threshold=0.5, debug=False):
    """IOU between image and mask

    Args:
        saliency (numpy array): [saliency map]
        ground_truth_mask (numpy array): [mask]
        threshold (float, optional): [threshold between values of
                                    saliency maps to consider as activated]. Defaults to 0.5.
        debug (bool, optional): [plot intermediate iages]. Defaults to True.

    Returns:
        IOU float: metrics
    """
    assert np.size(saliency) == np.size(ground_truth_mask)
    ground_truth_mask_area = np.where(ground_truth_mask > 0, 1, 0)
    saliency = min_max_tensor(saliency.astype("float"))
    saliency_area = np.where(saliency > threshold, 1, 0)
    if debug:
        plt.imshow(saliency_area)
        plt.title("Predicted area")
        plt.show()
        plt.imshow(ground_truth_mask_area)
        plt.title("mask area")
        plt.show()
    intersection = np.sum(saliency_area * ground_truth_mask_area)
    if debug:
        plt.imshow(saliency_area * ground_truth_mask_area)
        plt.title("Intersection")
        plt.show()
        plt.imshow(np.clip(ground_truth_mask_area + saliency_area, 0, 1))
        plt.title("Union")
        plt.show()
    union = np.sum(np.clip(ground_truth_mask_area + saliency_area, 0, 1))
    return intersection / union


def remove_top_features(
    saliency, image, percentage=0.05, background="Blured", insertion=False, summing=[]
):
    """[Remove top activated pixels]

    Args:
        saliency (numpy array): saliency map
        image (numpy array or pythorch tensor): image where to remove pixels
        percentage (float, optional): prcentage of pixels to remove. Defaults to 0.05.

    Returns:
        masked image: image with removed pixels
    """
    sorted_saliency = np.flip(np.sort(saliency.reshape(1, -1)))
    sorted_saliency_order = np.flip(np.argsort(saliency.reshape(1, -1)))
    threshold = sorted_saliency[0, int(percentage * len(sorted_saliency[0])) - 1]
    threshold_order = sorted_saliency_order[
        0, int(percentage * len(sorted_saliency_order[0])) - 1
    ]
    if background == "Blured":
        if not insertion:
            mask = 1 - np.where(saliency >= threshold, 0, 1)
            percentage = np.sum(np.where(saliency >= threshold, 1, 0))
            summing.append(np.sum(mask))
        else:
            mask = np.where(saliency >= threshold, 0, 1)
            percentage = np.sum(np.where(saliency >= threshold, 0, 1))
        kernel = GaussianLayer(sigma=100)
        masked_image = kernel(image.clone().detach(), mask=mask)
    elif background == "Black":
        if not insertion:
            mask = np.where(saliency >= threshold, 0, 1)
            percentage = np.sum(np.where(saliency >= threshold, 1, 0))
        else:
            mask = np.where(saliency >= threshold, 1, 0)
            percentage = np.sum(np.where(saliency >= threshold, 0, 1))
        if mask.shape < image.shape:
            mask = np.tile(mask, (3, 1, 1))
        if isinstance(image, np.ndarray):
            masked_image = image * mask
        else:
            masked_image = image * torch.from_numpy(mask)
    elif background == "Gray":
        if not insertion:
            mask = np.where(saliency >= threshold, 0, 1)
            gray = np.where(mask == 0, 128, 0)
        else:
            mask = np.where(saliency >= threshold, 1, 0)
            gray = np.where(mask == 0, 128, 0)
        if mask.shape < image.shape:
            mask = np.tile(mask, (3, 1, 1))
            gray = np.tile(gray, (3, 1, 1))
        if isinstance(image, np.ndarray):
            masked_image = (image * mask) + gray
        else:
            masked_image = (
                image * torch.from_numpy(mask).to(DEVICE)
            ) + torch.from_numpy(gray).to(DEVICE)
    return masked_image, percentage, summing


def remove_top_features_by_region(
    saliency,
    image,
    segments,
    segment,
    background="Blured",
    region=None,
    mask=None,
    insertion=False,
):
    """[Remove top activated pixels]

    Args:
        saliency (numpy array): saliency map
        image (numpy array or pythorch tensor): image where to remove pixels
        percentage (float, optional): prcentage of pixels to remove. Defaults to 0.05.

    Returns:
        masked image: image with removed pixels
    """
    if background == "Blured":
        if not insertion:
            loop_mask = np.where(segments == segment, 1, 0)

        else:
            loop_mask = np.where(segments == segment, 0, 1)
        if mask is not None:
            mask = np.clip(mask + loop_mask, 0, 1)

        else:
            mask = loop_mask
        kernel = GaussianLayer(sigma=100)
        masked_image = kernel(image.clone().detach(), mask=mask)
    elif background == "Black":
        if not insertion:
            loop_mask = np.where(segments == segment, 1, 0)
        else:
            loop_mask = np.where(segments == segment, 0, 1)
        if mask is not None:
            if mask is not None:
                mask = np.clip(1 - mask + loop_mask, 0, 1)
        else:
            mask = loop_mask
        if mask.shape < image.shape:
            mask = np.tile(mask, (3, 1, 1))
        if isinstance(image, np.ndarray):
            masked_image = image * mask
        else:
            masked_image = image * torch.from_numpy(mask).to(DEVICE)
    elif background == "Gray":
        # print(np.unique(segments))
        if not insertion:
            loop_mask = np.where(segments == segment, 0, 1)
        else:
            loop_mask = np.where(segments == segment, 1, 0)
        if mask is not None:
            if mask is not None:
                mask = np.clip(1 - mask + loop_mask, 0, 1)
            gray = np.where(mask == 0, 128, 0)
        else:
            mask = loop_mask
            gray = np.where(mask == 0, 128, 0)
        if mask.shape < image.shape:
            mask = np.tile(mask, (3, 1, 1))
            gray = np.tile(gray, (3, 1, 1))
        if isinstance(image, np.ndarray):
            masked_image = image * mask + gray
        else:
            masked_image = image * torch.from_numpy(mask).to(DEVICE) + torch.from_numpy(
                gray
            ).to(DEVICE)
    return masked_image, mask


def do_prediction(model, image, calibrated):
    # Apply model to input image
    if type(calibrated) is str:
        scores = nn.Softmax(dim=-1)(model(image.unsqueeze(0)))
    else:
        ypred = torch.log(nn.Softmax(dim=-1)(model(image.unsqueeze(0).float())))
        S_ = torch.hstack((ypred, torch.ones((len(ypred), 1)).to(DEVICE)))
        ypred = torch.mm(
            S_, torch.FloatTensor(calibrated.weights.transpose()).to(DEVICE)
        )
        scores = nn.Softmax(dim=-1)(ypred)
        # scores = nn.Softmax(dim=-1)(model(image.unsqueeze(0)) * calibrated._weights[0])
    return scores


def save_auc_non_random(calibrated, file, name, scores_list):
    # Save auc arrays
    if type(calibrated) is not str:
        with open(
            "code_final/food101/{}/{}/{}/auc_vectors_{}/auc_{}.npy".format(
                WDR, NAME_MODEL, file, "non_calibrated", name
            ),
            "wb",
        ) as f:
            np.save(f, scores_list)
    else:
        with open(
            "code_final/food101/{}/{}/{}/auc_vectors_{}/auc_{}.npy".format(
                WDR, NAME_MODEL, file, "calibrated", name
            ),
            "wb",
        ) as f:
            np.save(f, scores_list)


def save_auc_random(calibrated, file, name, scorelistinterp):
    if type(calibrated) is not str:
        with open(
            "code_final/food101/{}/{}/{}/auc_vectors_random/auc_{}.npy".format(
                WDR, NAME_MODEL, file, "calib_" + str(name)
            ),
            "wb",
        ) as f:
            np.save(f, scorelistinterp)
    else:
        with open(
            "code_final/food101/{}/{}/{}/auc_vectors_random/auc_{}.npy".format(
                WDR, NAME_MODEL, file, "non_calib" + str(name)
            ),
            "wb",
        ) as f:
            np.save(f, scorelistinterp)


def inside_loop_informations(score, true_prediction, i, do_print=False):
    if do_print:
        # print(score)
        print(score.argmax(), score.max())
        print("at {} score : {}".format(i, score[0, true_prediction]))
    else:
        pass


def plot_auc_scores(scores_list, plot):
    if plot:
        plt.plot(scores_list[0])
        plt.ylim([0, 1])
        plt.title("Predicted scores while removing most relevant pixels")
        plt.xlabel("% pixel removed")
        plt.ylabel("Scores")
        plt.show()


@use_gpu
def deletion(
    model,
    image,
    saliency_map,
    plot=True,
    auc_value=True,
    background="Blured",
    insertion=False,
    gpu=True,
    end=False,
    name=None,
    calibrated="calibrated",
    file=None,
):
    """Deletion experience. Remove pixel % by % and mesure the drop in
        the model accuracy

    Args:
        model (pytorch module): model
        image (pytorch tensor): input image
        saliency_map (numpy array): saliency maps
        plot (bool, optional): plot graphic showing drop in model accuracy. Defaults to True.
        auc_value (bool, optional): return area under curve. Defaults to True.

    Returns:
        auc_value :  area under curve , only if auc_value is True
    """
    # Apply model to input image
    scores = do_prediction(model, image, calibrated)
    # Get model's prediction
    true_prediction = scores.argmax().cpu().detach().numpy()
    prediction = true_prediction
    i = 1
    borne_max = 100
    print(borne_max)
    # Put prediction as first element of score list
    scores_list = np.zeros((1, borne_max))
    scores_list[0, 0] = scores.detach()[0, true_prediction]
    # Remove pixels by percentage accorting to heatmap
    percentage = 0
    summing = []
    while i < borne_max:
        # generate masked image
        masked_image, _, summing = remove_top_features(
            saliency_map,
            image,
            percentage=i / borne_max,
            background=background,
            insertion=insertion,
            summing=summing,
        )
        # Apply model to masjed image
        if gpu:
            masked_image.to(DEVICE)
        score = do_prediction(model, masked_image, calibrated)
        inside_loop_informations(score, true_prediction, i, do_print=False)
        # Append score to score list
        scores_list[0, i] = score[0, true_prediction]
        i += 1
    # Save score array where needed
    save_auc_non_random(calibrated, file, name, scores_list)
    # plot score
    plot_auc_scores(scores_list, plot)
    # compute auc
    if auc_value:
        auc = metrics.auc(range(borne_max), scores_list[0])
        return auc


def generate_random_masks(image, scores):
    np_image = np.array(transforms.ToPILImage()(image.squeeze()))
    less = 0
    segments = np.arange(101)
    while len(np.unique(segments)) > 99:
        segments = slic(np_image, n_segments=99 - less, max_iter=2, start_label=0)
        less += 10
    # print("class :", true_prediction)
    up = scores.max().detach()
    seg = np.unique(segments)
    np.random.shuffle(seg)
    return seg, segments


def interpolate_to_100(scores_list):
    scorelistinterp = np.clip(
        np.interp(
            np.arange(100), np.linspace(0, 100, len(scores_list[0])), scores_list[0]
        ),
        0,
        1,
    )
    return scorelistinterp


@use_gpu
def deletion_random(
    model,
    image,
    saliency_map,
    plot=True,
    auc_value=True,
    background="Blured",
    insertion=False,
    gpu=True,
    end=True,
    name=None,
    calibrated="calibrated",
    file=None,
    seed=0,
):
    """Deletion experience. Remove pixel % by % and mesure the drop in
        the model accuracy random pixel by region. Oversegment input an remove region by region.

    Args:
        model (pytorch module): model
        image (pytorch tensor): input image
        saliency_map (numpy array): saliency maps
        plot (bool, optional): plot graphic showing drop in model accuracy. Defaults to True.
        auc_value (bool, optional): return area under curve. Defaults to True.

    Returns:
        auc_value :  area under curve , only if auc_value is True
    """

    scores_list = np.zeros((1, 100))
    scores = do_prediction(model, image, calibrated)
    true_prediction = scores.argmax().cpu().detach().numpy()
    prediction = true_prediction
    i = 0
    scores_list[0, 0] = scores.detach()[0, true_prediction]
    scorelistinterplist = []
    for i in range(5):
        with RandomManager(5):
            seg, segments = generate_random_masks(image, scores)
            # print(seg)
        scores_list = scores_list[:, : len(np.unique(segments)) + 1]
        mask = None
        masked_image = image
        while i < len(np.unique(segments)):
            try:
                segi = seg[i]
                masked_image, mask = remove_top_features_by_region(
                    saliency_map,
                    masked_image,
                    segments,
                    segment=segi,
                    background=background,
                    mask=mask,
                    insertion=insertion,
                )
                """heatmap = transforms.ToPILImage()(masked_image)
                plt.imshow(heatmap)
                plt.show()"""
                if gpu:
                    masked_image.to(DEVICE)
                score = do_prediction(model, masked_image, calibrated)
                # print("at {} score : {}".format(i, score[0, true_prediction]))
                scores_list[0, i + 1] = score.detach()[0, true_prediction]
                i += 1
            except Exception as e:
                print(e)
                i += 1
                break
        scorelistinterp = interpolate_to_100(scores_list)
        scorelistinterplist.append(scorelistinterp)
    assert len(scorelistinterplist) == 5
    assert len(np.array(scorelistinterplist)) == 5
    save_auc_random(calibrated, file, name, np.array(scorelistinterplist))
    plot_auc_scores(scores_list, plot)
    if auc_value:
        auc = metrics.auc(range(100), scorelistinterp)
        return auc


def energy_based_clicking_game(saliency, ground_truth_mask):
    """Energy based metric (percentage enrgy of saliency
    present in mask vs total energy)
    Args:
        saliency (numpy array): saliency map
        ground_truth_mask (numpy array): ground truth (segmented image)

    Returns:
        energy : float ratio energy
    """
    assert np.size(saliency) == np.size(ground_truth_mask)
    ground_truth_mask_area = np.where(ground_truth_mask > 0, 1, 0)
    energy = ground_truth_mask_area * saliency
    return np.sum(energy) / np.sum(saliency)


if __name__ == "__main__":
    array = np.arange(9, 18).reshape(3, 3)
    mask = np.zeros((3, 3))
    mask[0, 0], mask[0, 1] = 1, 1
    """true_array = array
    true_array[0, 0], true_array[0, 1] = 0, 0
    assert np.allclose(
        remove_top_features(mask, array, percentage=0.99, background="Black"),
        true_array,
    )"""
    true_array_first, true_array_second = array, array
    true_array_first[0, 0] = 1
    true_array_second[0, 1] = 1
    assert np.allclose(
        remove_top_features(mask, array, percentage=0.49, background="Black"),
        true_array_first,
    ) or np.allclose(
        remove_top_features(mask, array, percentage=0.49, background="Black"),
        true_array_second,
    )
