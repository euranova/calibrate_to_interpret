import xml.etree.ElementTree as ET
import os
import pickle

import numpy as np
import torch

import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torchvision
import random

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


class DatasetPreprocessorCifar100:
    """
    Dataset processor object
    """

    def __init__(self):
        """
            Initialisation get classes name and corresponding plae in the output vector
        Args:
            None
        Returns:
            None
        """
        """self.classes = benchmark.datasets.VOC_CLASSES
        self.corresps = {}
        for class_i in range(len(self.classes)):
            self.corresps[self.classes[class_i]] = class_i"""

    def load_files(
        self,
        for_segmentation=False,
        image_path="../dataset/VOCdevkit/VOC2007/JPEGImages",
        annotation_path="../dataset/VOCdevkit/VOC2007/Annotations",
        mask_path="../dataset/VOCdevkit/VOC2007/SegmentationClass",
        save=False,
        limit=None,
        label_as_int=True,
        calibration_set=True,
    ):
        """
            Load previously downloaded files (torchvision format)
            First parse the xml files , get name and object present in the image
            If for_segmentation then get the according segmentation image and binarize it
            return X the image (pytorch tensor),Y the labels (float), the filenames for visualisation and if needed the masks
        Args:
            for_segmentation: bool, if true then return the segmentation masks
            image_path: str, path where the image are (jpg format)
            annotation_path: str, path where the annotations are (xml format)
            mask_path: str, path where the masks are (png format )
            save: bool, if true save the dataset in a dataset folder (NEED DATASETS FOLDER TO EXIST)
            limit: int, to limit exportation at a certain number of images
        Returns:
            X: list of pytorch tensor of image (3 dimensions)
            Y: list of (!!!) float , labels
            filenames, name of image files in the same order as X
            masks, list of np.array , list of masks returned only if for_segmentation is True
        """
        if for_segmentation:
            files = os.listdir(mask_path)
        else:
            files = os.listdir(annotation_path)
            files_seg = os.listdir(mask_path)
        Y, X = [], []
        masks, filenames = [], []
        if calibration_set:
            files = files[int(len(files) / 2) :]
        else:
            files = files[: int(len(files) / 2)]
        for i, file in enumerate(files):
            if file == files_seg:
                break
            classes, filename = self.parse_xml_line(file, annotation_path)
            img_path = os.path.join(image_path, filename)
            image = self.prepare_data(path_img=img_path)
            X.append(image)
            try:
                if label_as_int:
                    Y.append(int(self.corresps.get(classes[0])))
                else:
                    Y.append(classes[0])
            except Exception:
                if label_as_int:
                    Y.append(int(self.corresps.get(classes)))
                else:
                    Y.append(classes)
            filenames.append(img_path)
            if for_segmentation:
                mask = self.get_mask_from_img(mask_path, filename)
                masks.append(mask)
            if limit is not None:
                if i == limit:
                    Y = Y[: len(X)]
                    break
        if save:
            self.save_files(X, Y, masks, filenames, for_segmentation)
        if for_segmentation:
            return X, np.array(Y), filenames, masks
        return X, np.array(Y), filenames

    def parse_xml_line(self, file, annotation_path):
        file = file.replace(".png", ".xml")
        tree = ET.parse(os.path.join(annotation_path, file))
        root = tree.getroot()
        filename = root.find("filename").text
        classes = []
        for obj in root.findall("object"):
            classe = obj.find("name").text
            classes.append(classe)
        return classes, filename

    def save_files(self, X, Y, masks, filenames, for_segmentation, path="../dataset"):
        """
        Save files in desired folders

        """
        with open(os.path.join(path, "X"), "wb") as output_file:
            pickle.dump(X, output_file)
        with open(os.path.join(path, "Y"), "wb") as output_file:
            pickle.dump(Y, output_file)
        if for_segmentation:
            with open(os.path.join(path, "masks"), "wb") as output_file:
                pickle.dump(masks, output_file)
        with open(os.path.join(path, "filenames"), "wb") as output_file:
            pickle.dump(filenames, output_file)

    def load_files_by_batch(
        self,
        for_segmentation=False,
        batch_size=100,
        image_path="../dataset/VOCdevkit/VOC2007/JPEGImages",
        annotation_path="../dataset/VOCdevkit/VOC2007/Annotations",
        mask_path="../dataset/VOCdevkit/VOC2007/SegmentationClass",
        label_as_int=True,
        calibration_set=True,
    ):
        """
            Same as load_files but as generator (return batch_size objects at each iteration)
        Args:
            for_segmentation: bool, if true then return the segmentation masks
            image_path: str, path where the image are (jpg format)
            annotation_path: str, path where the annotations are (xml format)
            mask_path: str, path where the masks are (png format )
            save: bool, if true save the dataset in a dataset folder (NEED DATASETS FOLDER TO EXIST)
        Returns:
            X: list of pytorch tensor of image (3 dimensions)
            Y: list of (!!!) float , labels
            filenames, name of image files in the same order as X
            masks, list of np.array , list of masks returned only if for_segmentation is True
        """
        if for_segmentation:
            files = os.listdir(mask_path)
        else:
            files = os.listdir(annotation_path)
            files_seg = os.listdir(mask_path)
        if calibration_set:
            files = files[: int(len(files) / 2)]
        else:
            files = files[int(len(files) / 2) :]
        for i in range(0, len(files) - batch_size - 1, batch_size):
            if for_segmentation:
                files = os.listdir(mask_path)
            else:
                files = os.listdir(annotation_path)
                files_seg = os.listdir(mask_path)
            files = files[i : i + batch_size]
            Y, X = np.zeros((len(files))), []
            masks, filenames = [], []
            for i, file in enumerate(files):
                if file == files_seg:
                    break
                classes, filename = self.parse_xml_line(file, annotation_path)
                img_path = os.path.join(image_path, filename)
                image = self.prepare_data(path_img=img_path)
                X.append(image)
                if label_as_int:
                    Y[i] = int(self.corresps.get(classes[0]))
                else:
                    Y[i] = classes[0]
                # Y[i] = int(self.corresps.get(classes[0]))
                filenames.append(img_path)
                if for_segmentation:
                    mask = self.get_mask_from_img(mask_path, filename)
                    masks.append(mask)
            if for_segmentation:
                yield X, Y, filenames, masks
            yield X, Y, filenames

    def prepare_data(self, path_img, dataset="voc"):
        """
            Adapt image into a format of torchray models
        Args:
            path_img: str path to the imput image
        Returns:
            image: Pytorch tensor
        """
        image = Image.open(path_img)

        transform_ = transforms.Compose(
            [
                # transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.2471, 0.2435, 0.2616), (0.25, 0.25, 0.25)),
            ]
        )

        image = transform_(image)
        self.shape = image.shape
        return image

    def get_mask_from_img(self, path, img_path):
        """
        Get mask from masks folder

        """
        img_path = img_path.replace(".jpg", ".png")
        image = Image.open(os.path.join(path, img_path))
        image = image.resize((self.shape[-1], self.shape[-2]), Image.BILINEAR)
        mask = np.array(image)
        return mask

    def download_dataset_voc(
        self, root="../dataset", year="2007", image_set="test", download=True
    ):
        dataset = torchvision.datasets.VOCDetection(
            root=root, year=year, image_set=image_set, download=download
        )
        return dataset

    def download_dataset_cifar(
        self, root="../cifar100", train_set=False, download=True
    ):
        dataset = torchvision.datasets.CIFAR100(
            root=root,
            train=train_set,
            transform=None,
            target_transform=None,
            download=download,
        )
        return dataset

    def get_item(
        self,
        for_segmentation=False,
        image_path="../cifar10/VOCdevkit/VOC2007/JPEGImages",
        annotation_path="../dataset/VOCdevkit/VOC2007/Annotations",
        mask_path="../dataset/VOCdevkit/VOC2007/SegmentationClass",
        save=False,
        limit=None,
        calibration_set=True,
    ):
        print(os.getcwd())
        if for_segmentation:
            files = os.listdir(mask_path)
        else:
            files = os.listdir(annotation_path)
        if calibration_set:
            files = files[: int(len(files) / 2)]
        else:
            files = files[int(len(files) / 2) :]
        sample = np.random.randint(low=0, high=len(files))
        file = files[sample]
        Y, X = np.zeros((1)), []
        filenames, masks = [], []
        classes, filename = self.parse_xml_line(file, annotation_path)
        img_path = os.path.join(image_path, filename)
        image = self.prepare_data(path_img=img_path)
        X.append(image)
        Y[0] = int(self.corresps.get(classes[0]))
        filenames.append(img_path)
        if for_segmentation:
            mask = self.get_mask_from_img(mask_path, filename)
            masks.append(mask)
        if for_segmentation:
            return X[0], Y[0], filenames[0], masks[0]
        return X[0], Y[0], filenames[0]
