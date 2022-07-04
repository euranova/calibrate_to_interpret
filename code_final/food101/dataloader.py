import numpy as np
import os
import pandas as pd
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import random
from torch.utils.data import Dataset, DataLoader


class Dataloader_food(Dataset):
    def __init__(self, input_size=(224, 224), mode="test"):
        # super(self).__init__()
        self.mode = mode
        self.file = "code_final/food101/meta/" + mode + ".txt"
        print(self.file)
        images_names = []
        self.input_size = input_size

        with open("code_final/food101/meta/classes.txt", "r") as txt:
            self.classes = [l.strip() for l in txt.readlines()]
        with open(self.file) as directory:
            for name in directory:
                images_names.append(name)
        print(len(images_names))
        random.shuffle(images_names)
        self.images_names = images_names

    def __tranforms__():
        pass

    def __repr__(self):
        return "Food Dataloader in mode {} ".format(self.mode)

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        image_name = self.images_names[idx].replace("\n", "")
        try:
            img = Image.open(
                os.path.join("code_final/food101/images/", image_name + ".jpg")
            )
        except Exception as e:
            print(e)
        transform = transforms.Compose(
            [
                # resize it to the size indicated by image_size
                transforms.Resize((224, 224)),
                # convert it to a tensor
                transforms.ToTensor(),
                # normalize it to the range [−1, 1]
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        img = transform(img)
        label = int(
            self.classes.index(
                image_name.replace("code_final/food101/meta/", "").split("/")[0]
            )
        )
        return np.array(img), label
