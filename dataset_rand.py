# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
from cvtransforms import *
import torch
import glob
import re
import copy
import json
import random
import editdistance
import csv
#from extract_lips import get_lips



class MyDataset(Dataset):
    labels = ["ama", "normal", "sexy", "tsun", "whis"]


    def __init__(self, anno_path, file_list, vid_pad, phase):
        self.anno_path = anno_path # 正解文字列データのパスを格納
        self.vid_pad = vid_pad # ビデオデータのパッディングの数
        self.phase = phase # train か testのフェーズを格納
        self.rand_data = []
        # ビデオまでのパスを読み出す
        with open(os.path.join(file_list), 'r') as f:
            buf = (f.read()).split('\n')
            while '' in buf:
                buf.remove('')
            for b in buf:
                b = b.replace("lips", "rand_label")
                self.rand_data.append(b+'.csv')

    # データのロード
    def __getitem__(self, idx):
        rand = self.rand_data[idx] # spkビデオの入っているフォルダまでのパス
        path_sep = rand.split('/')
        rand = self._load_rand(rand) # ビデオデータのフレームをロード
        label = self._load_label(path_sep[-2])
        rand = self._rand_padding(rand, self.vid_pad) # ビデオのパッディング
        return {
            'rand': torch.FloatTensor(rand),
            'label': torch.FloatTensor(label),
        }

    # データの長さを返す関数
    def __len__(self):
        return len(self.video_data)
    
    # ビデオのロード
    def _load_rand(self, p):
        rand = []
        with open(p, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                row.remove('')
                number = list(map(float, row))
                number = np.array(number)
                number = number.reshape(number.shape[0]//2, 2)
                number[:, 1] *= -1
                rand.append(number)
        rand = np.stack(rand, axis=0).astype(np.float32)
        return rand
    
    def _load_label(self, kind):
        result = np.zeros(5)
        result[MyDataset.labels.index(kind)] = 1
        return result

    def _rand_padding(self, array, length):
        size = length - array.shape[0]
        pd_arr = np.zeros([size, array.shape[1], array.shape[2]])
        return np.vstack([array, pd_arr])

    # パッディング
    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)


import options as opt
import matplotlib.pyplot as plt
if __name__ == "__main__":
    dataset = MyDataset(
            opt.anno_path,
            opt.val_list,
            opt.vid_padding,
            'train')
    data = (dataset.__getitem__(0)).get('rand')
    sample = data[0]
    print(data.size())
    print(sample.size())
    print(dataset.__getitem__(0).get('label'))
    plt.scatter(sample[:, 0], sample[:, 1])
    plt.show()
