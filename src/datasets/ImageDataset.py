import torch
from torch.utils import data
import numpy as np
import os

class ImageDataset(data.Dataset):
    def __init__(self, num_instances, data_part, data_dir="..."):
        # path_original = os.path.join(data_dir, f"{data_part}/original")
        # path_lowres = os.path.join(data_dir, f"{data_part}/low_resolution")
        path_original = os.path.join(data_dir, f"ground_truth")
        path_lowres = os.path.join(data_dir, f"input")

        self.shape_original = self.get_shape(path_original)
        self.shape_lowres = self.get_shape(path_lowres)
        self.images_original = torch.zeros((num_instances, *self.shape_original))
        self.images_low_resolution = torch.zeros((num_instances, *self.shape_lowres))
        
        # chargement des fichiers .npy
        for n in range(num_instances):
            file_O = os.path.join(path_original, f"{n}.npy")
            file_LR = os.path.join(path_lowres, f"{n}.npy")

            # chargement et conversion
            O = np.load(file_O).astype("float32")
            LR = np.load(file_LR).astype("float32")

            O = self.normalize_image(O)
            LR = self.normalize_image(LR)
            
            # reshape et conversion en Tensor
            self.images_original[n] = torch.from_numpy(O.reshape(self.shape_original))
            self.images_low_resolution[n] = torch.from_numpy(LR.reshape(self.shape_lowres))

    def __getitem__(self, index):
        img_orig = self.images_original[index]
        img_lowres = self.images_low_resolution[index]
        return img_orig, img_lowres

    def __len__(self):
        return len(self.images_original)

    def get_shape(self, path):
        first_file = os.path.join(path, "0.npy")
        sample = np.load(first_file).astype("float32")
        if len(sample.shape) == 2:
            sample = sample[np.newaxis, :, :]
        return sample.shape

    def normalize_image(img):
        mini = torch.min(img)
        maxi = torch.max(img)
        normalized = (img - mini)/(maxi - mini)
        return normalized