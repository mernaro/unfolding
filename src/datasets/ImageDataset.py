import torch
from torch.utils import data
import numpy as np
import os
import pandas as pd

class ImageDataset(data.Dataset):
    def __init__(self, num_instances, data_part, data_dir="..."):
        path_original = os.path.join(data_dir, f"{data_part}/ground_truth")
        path_lowres = os.path.join(data_dir, f"{data_part}/input")
        path_params = os.path.join(data_dir, f"{data_part}/params.csv")

        self.params_df = pd.read_csv(path_params)
        self.items = []
        
        # chargement des fichiers .npy
        for n in range(num_instances):
            file_O = os.path.join(path_original, f"{n}.npy")
            file_LR = os.path.join(path_lowres, f"{n}.npy")

            # chargement et conversion
            O = np.load(file_O).astype("float32")
            LR = np.load(file_LR).astype("float32")

            O = self.normalize_image(O)
            LR = self.normalize_image(LR)

            row = self.params_df.iloc[n]
            params_tensor = torch.tensor([
                row["blur_size"] if not np.isnan(row["blur_size"]) else 0,
                row["blur_sigma"] if not np.isnan(row["blur_sigma"]) else 0,
                row["decimation"] if not np.isnan(row["decimation"]) else 0,
                row["noise_value"] if not np.isnan(row["noise_value"]) else 0,
                row["noise_db"] if not np.isnan(row["noise_db"]) else 0,
                ], dtype=torch.float32)
            
            # reshape et conversion en Tensor
            self.items.append((torch.from_numpy(O), torch.from_numpy(LR), params_tensor))

    def __getitem__(self, index):
        img_orig = self.items[index][0]
        img_lowres = self.items[index][1]
        param = self.items[index][2]
        return img_orig, img_lowres, param

    def __len__(self):
        return len(self.items)

    def get_shape(self, path):
        first_file = os.path.join(path, "0.npy")
        sample = np.load(first_file).astype("float32")
        if len(sample.shape) == 2:
            sample = sample[np.newaxis, :, :]
        return sample.shape

    def normalize_image(self, img):
        mini = np.min(img)
        maxi = np.max(img)
        normalized = (img - mini)/(maxi - mini)
        return normalized
        
def get_batch_with_variable_size_image(batch):
    imgs_input = []
    imgs_ground_truth = []
    params_list = []

    for elem in batch:
        imgs_input.append(elem[0])
        imgs_ground_truth.append(elem[1])
        params_list.append(elem[2])

    # Your custom processing here
    return imgs_input, imgs_ground_truth, params_list