import numpy as np
import torch


def confidences_from_scores(scores_matrix, predictions, model):
    # Turning predictions into a numerical usable format
    return np.max(scores_matrix, axis=1)

    # Old version
    #corresps = {}
    #for class_i in range(len(model.classes_)):
    #    corresps[model.classes_[class_i]] = class_i
    #predictions_num = np.vectorize(corresps.get)(predictions)

    # Gathering scores relative to predicted class
    #scores_predicted_class = []
    #for i in range(len(predictions)):
    #    scores_predicted_class.append(scores_matrix[i, predictions_num[i]])

    #return np.array(scores_predicted_class)


def get_pytorch_model_scores(model, supervised_data_loader, device):
    """
    Computes output scores batch by batch.
    Args:
        model: nn.Module, the model of interest.
        supervised_data_loader: a dataloader which outputs an (X, y) tuple at each iteration.
        device: pointer to device on which computations should be done.

    Returns:
    The scores matrix as an np.ndarray.
    """

    # model
    assert isinstance(model, torch.nn.Module)

    # supervised_data_loader
    assert isinstance(supervised_data_loader, torch.utils.data.DataLoader)

    # device
    # TODO assert type(device) is <>
    print(f"Add assertion : device has to be a {type(device)}")

    scores = []
    with torch.no_grad():
        for x_batch, y_batch in iter(supervised_data_loader):
            scores.append(model(x_batch.to(device)).detach())
    return torch.cat(scores).cpu().numpy()
