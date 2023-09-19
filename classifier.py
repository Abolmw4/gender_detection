#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Abolfazl Asghari
# Email: a.ashgari251@gmail.com
# Created Date: 31 May 2023
# version ='1.0'
# ---------------------------------------------------------------------------
""" This program takes an absolute directory path as input and separate its
images, then moves them to separate directories."""
# ---------------------------------------------------------------------------
from __future__ import annotations
import time
import numpy as np
import torch
from torchvision.transforms import transforms
import cv2
import os
import shutil
from tqdm import tqdm


class GenderSeperator:
    def __init__(self, source_directory: str, destination_directory: str, model_path: str) -> None:
        self.source_directory = source_directory
        self.model_path = model_path
        self.destination_directory = destination_directory
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
        self.valid_images = list()
        self.invalid_images = list()

    @property
    def source_directory(self) -> str:
        return self._source_directory

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def destination_directory(self) -> str:
        return self._destination_directory

    @source_directory.setter
    def source_directory(self, value: str) -> None:
        if os.path.isdir(value):
            self._source_directory = value
        else:
            raise ValueError("input value must be directory address and must be absolute path!")

    @model_path.setter
    def model_path(self, value) -> None:
        if value.find(".pth") != -1 or value.find(".pt") != -1:
            self._model_path = value
        else:
            raise ValueError("input value must be .pth file or .pt file and must be absolute path!")

    @destination_directory.setter
    def destination_directory(self, value) -> None:
        if os.path.isdir(value):
            self._destination_directory = value
        else:
            raise ValueError("input value must be directory address and must be absolute path!")

    def __call__(self, device) -> None:
        device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        print(f"your device is {device}\n")
        net = torch.jit.load(self.model_path).to(device)
        net.eval()

        os.mkdir(f'{self.destination_directory}/Male') if 'Male' not in os.listdir(
            self.destination_directory) else print(f"{os.path.join(self.destination_directory, 'Male')} exist!")
        os.mkdir(f"{self.destination_directory}/Female") if 'Female' not in os.listdir(
            self.destination_directory) else print(f"{os.path.join(self.destination_directory, 'Female')} exist!")
        os.mkdir(f"{self.destination_directory}/Unknown") if 'Unknown' not in os.listdir(
            self.destination_directory) else print(f"{os.path.join(self.destination_directory, 'Unknown')} exist!")

        while True:
            if not os.listdir(self.source_directory):
                print("Please wait for 5 seconds until the directory is filled.")
                time.sleep(5)
            for image in os.listdir(self.source_directory):
                if image.find(".jpg") != -1 or image.find(".jpeg") != -1 or image.find(".png") != -1 or image.find(
                        ".JPG") != -1 or image.find(".JPEG") != -1 or image.find(".PNG") != -1:
                    self.valid_images.append(os.path.join(self.source_directory, image))
                else:
                    print(f"file {os.path.join(self.source_directory, image)} is not image!")
                    self.invalid_images.append(os.path.join(self.source_directory, image))
                    continue

            for item in tqdm(self.valid_images, desc="Transferring", leave=True):
                try:
                    image = cv2.imread(item, 1)
                    image_for_model = self.transform(image).to(device)
                except Exception as e:
                    print(f"{e}")
                    print(f"the file {item} is an Image") if isinstance(image, np.ndarray) else print(
                        f"the file {item} Not an image")
                    continue
                image_for_model = image_for_model.reshape(
                    (1, image_for_model.shape[0], image_for_model.shape[1], image_for_model.shape[2]))
                output = net(image_for_model)
                if output[0, 0] > output[0, 1] and (output[0, 0] * 100 - output[0, 1] * 100) > 27.0:
                    try:
                        shutil.move(item, os.path.join(self.destination_directory, "Male"))
                    except Exception as e:
                        print(f"{e}")
                        continue
                        # new_str = item[item.rindex('/'):]
                        # os.remove(os.path.join(self.destination_directory, "Female", new_str))
                        # shutil.move(item, os.path.join(self.destination_directory, "Male", new_str))
                elif output[0, 1] > output[0, 0] and (output[0, 1] * 100 - output[0, 0] * 100) > 27.0:
                    try:
                        shutil.move(item, os.path.join(self.destination_directory, "Female"))
                    except Exception as e:
                        print(f"{e}")
                        continue
                        # new_str = item[item.rindex('/'):]
                        # os.remove(os.path.join(self.destination_directory, "Female", new_str))
                        # shutil.move(item, os.path.join(self.destination_directory, "Female", new_str))
                else:
                    try:
                        shutil.move(item, os.path.join(self.destination_directory, "Unknown"))
                    except Exception as e:
                        print(f"{e}")
                        continue
                        # new_str = item[item.rindex('/'):]
                        # os.remove(os.path.join(self.destination_directory, "Female", new_str))
                        # shutil.move(item, os.path.join(self.destination_directory, "Unknown", new_str))
            self.valid_images.clear()

    def __repr__(self) -> str:
        gender_seperator_str: str = f"images_directory: {self.source_directory}\n" \
                                    f"model_path: {self.model_path}\n" \
                                    f"your device available: {'cuda' if torch.cuda.is_available() else 'cpu'}\n" \
                                    f"invalid images: {self.invalid_images}\n"
        return gender_seperator_str
