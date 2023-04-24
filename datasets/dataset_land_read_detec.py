import numpy as np
import os
from torch.utils.data import Dataset
import torch
import csv
#from extract_lips import get_lips

class MyDataset(Dataset):
    labels = ["ama", "recitation", "tsun", "normal", "sexy"]

    def __init__(self, dataroot, datafile, rand_pad, rand_mean, rand_std, input_mode=0):
        """
            dataroot: path of data file.
            datafile: this is data filename.
            rand_pad: randmark data padding num.
            rand_mean: rand_mean[0] -> mean of x, rand_mean[1] -> mean of y
            rand_std: rand_std[0] -> std of x, rand_std[1] -> std of y
            input_mode: 0 is norm, 1 is xy rand, 2 is xy0 - xy1
                3 is Z score -> 68 x (x, y), 
        """
        self.rand_pad = rand_pad
        self.input_mode = input_mode
        self.files = []
        self.rand_mean = rand_mean
        self.rand_std = rand_std
        with open(os.path.join(dataroot, datafile), "r") as f:
            self.files = [path.rstrip() for path in f.readlines()]
    
    def __getitem__(self, idx):
        data_path = self.files[idx]
        path_sep = data_path.split("\\")
        if "" in path_sep:
            path_sep.remove("")
        rand = self._load_rand(data_path)
        rand = self._rand_padding(rand, self.rand_pad)
        if self.input_mode == 0:
            rand = rand.reshape(1, rand.shape[-1])
        kind = self._load_label(path_sep[-2])
        if self.input_mode == 3:
            rand = rand.reshape(rand.shape[0], -1)
            # rand = rand.reshape(rand.shape[1], rand.shape[0])
        return {
            'rand': torch.FloatTensor(rand),
            'label': torch.LongTensor(kind),
        }
    
    def __len__(self):
        return len(self.files)
    
    def _load_rand(self, p):
        rand = []
        with open(p, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                row.remove('')
                number = list(map(float, row))
                number = np.array(number)
                number = number.reshape(number.shape[0]//2, 2)
                # number[:, 1] *= -1
                if self.input_mode > 0:
                    rand.append(number)
                elif self.input_mode == 0:
                    rand.append(np.linalg.norm(number[48:, :]))
        rand = np.array(rand)

        if self.input_mode == 0:
            rand_min = rand.min()
            rand_max = rand.max()
            rand = (rand - rand_min) \
                / (rand_max - rand_min)
        if self.input_mode == 1:
            rand_min_w = None
        if self.input_mode == 2:
            rand = [rand[i]-rand[i+1] for i in range(rand.shape[0]-1)]
            rand = np.stack(rand, axis=0).astype(np.float32)
        
        if self.input_mode == 3:
            rand[:, :, 0] -= self.rand_mean[0]
            rand[:, :, 0] /= self.rand_std[0]
            rand[:, :, 1] -= self.rand_mean[1]
            rand[:, :, 1] /= self.rand_std[1]
        return rand
    
    def _load_label(self, kind):
        result = np.zeros(1,)
        result[0] = MyDataset.labels.index(kind)
        return result
    
    def _rand_padding(self, array, length):
        size = length - array.shape[0]
        pd_arr = None
        if self.input_mode == 0:
            pd_arr = np.zeros([size,])
        else:
            pd_arr = np.zeros([size, array.shape[1], array.shape[2]])
        result = np.vstack([array, pd_arr]) if self.input_mode \
              else np.hstack(np.hstack([array, pd_arr]))
        return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = MyDataset(
        "data", "train_path.txt", 1112,
        [0, 0],
        [1, 1],
        3
    )
    lengs_min_w = 100000
    lengs_max_w = -100000
    lengs_min_h = 100000
    lengs_max_h = -100000
    for leng in range(dataset.__len__()):
        data = dataset.__getitem__(leng).get("rand")
        data = data.detach().numpy()
        exit()
        data[:, :, 0] += 100
        data[:, :, 1] += 200
        if data[:, :, 0].min() < lengs_min_w:
            lengs_min_w = data[:, :, 0].min()
        if data[:, :, 0].max() > lengs_max_w:
            lengs_max_w = data[:, :, 0].max()
        if data[:, :, 1].min() < lengs_min_h:
            lengs_min_h = data[:, :, 1].min()
        if data[:, :, 1].max() > lengs_max_h:
            lengs_max_h = data[:, :, 1].max()
    print(lengs_min_w, lengs_max_w)
    print(lengs_min_h, lengs_max_h)
