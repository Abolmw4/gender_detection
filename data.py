from torch.utils.data import Dataset
import os
import cv2
from sklearn.preprocessing import OneHotEncoder


class AbolfazlDataset(Dataset):
    def __init__(self, dataroot, image_transforms):
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.enc.fit([["male", "female"]])
        self.image_transforms = image_transforms
        self.data = list()
        for item in os.listdir(dataroot):
            if item == "male":
                for image in os.listdir(os.path.join(dataroot, item)):
                    male_data = list()
                    male_data.append(os.path.join(dataroot, item, image))
                    male_data.append(['male', 0])
                    self.data.append(male_data)
            else:
                for image in os.listdir(os.path.join(dataroot, item)):
                    female_data = list()
                    female_data.append(os.path.join(dataroot, item, image))
                    female_data.append([0, 'female'])
                    self.data.append(female_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image = cv2.imread(self.data[idx][0], 1)
        if self.image_transforms:
            image = self.image_transforms(image)
            label = self.enc.transform([self.data[idx][-1]]).toarray()
            lbl = label.reshape(-1)
        else:
            raise ValueError("label transform")
        return image, lbl
