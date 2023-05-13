# encoding: utf-8
import numpy as np
import cv2
import os
from torch.utils.data import Dataset
from cvtransforms import *
import torch
import editdistance
import options.options_lipnet_land_multi as opt
import csv
#from extract_lips import get_lips



class MyDataset(Dataset):
    def choice_letters(kind):
        if 'anno_data_romaji' == kind:
            return [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        return [' ', 'a', 'i', 'u', 'e', 'o', 'k', 's', 't', 'n', 'h', 'm', 'y', 'r', 'w', '*', 'g', 'z', 'd', 'b', 'p', 'l', 'v', '^', '-']
    # letters = [' ', 'A', 'I', 'U', 'E', 'O']
    letters = choice_letters(opt.anno_kind)
    labels = ["ama", "normal", "sexy", "tsun", "whis"]


    def __init__(self, video_path, anno_path, file_list, vid_pad, txt_pad, phase, txt_kind, color_mode, land_mean, land_std, land_input_mode=3):
        self.anno_path = anno_path # 正解文字列データのパスを格納
        self.vid_pad = vid_pad # ビデオデータのパッディングの数
        self.txt_pad = txt_pad # テキストのパッディング
        self.phase = phase # train か testのフェーズを格納
        self.land_input_mode = land_input_mode
        self.land_mean = land_mean
        self.land_std = land_std
        self.video_data = []
        self.txt_kind = txt_kind
        self.color_mode = color_mode
        # ビデオまでのパスを読み出す
        with open(os.path.join(video_path, file_list), 'r') as f:
            self.video_data = [os.path.join(video_path, line.rstrip()) for line in f.readlines()]
        self.video_path = video_path
    # データのロード
    def __getitem__(self, idx):
        vid = self.video_data[idx] # spkビデオの入っているフォルダまでのパス
        path_sep = vid.split('/')
        anno_kind = None
        land_p = None
        if (path_sep[5])[0] == 'R':                
            anno_kind = path_sep[:6]
            anno_kind.append(self.txt_kind)
            anno_kind.append(path_sep[-1]+'.txt')
            land_p = path_sep[:6]
            land_p.append('rand_label')
            land_p.append(path_sep[-1])
        else:
            if path_sep[-2] == 'recitation':
                number = (path_sep[-1])[-3:]
                anno_kind = path_sep[:6]
                anno_kind.append(self.txt_kind)
                anno_kind.append('recitation')
                anno_kind.append('recitation324_'+number+'.txt')
                land_p = path_sep[:7]
                land_p.append('rand_label')
                land_p += path_sep[-2:]
            else:
                number = (path_sep[-1])[-3:]
                anno_kind = path_sep[:6]
                anno_kind.append(self.txt_kind)
                anno_kind.append('emotion')
                anno_kind.append('emotion100_'+number+'.txt')
                land_p = path_sep[:7]
                land_p.append('rand_label')
                land_p += path_sep[8:]
        vid = self._load_vid(vid) # ビデオデータのフレームをロード
        
        anno_p = '/'.join(anno_kind)
        land_p = '/'.join(land_p)
        land_p += '.csv'
        
        land = self._load_land(land_p)
        land = self._land_padding(land, self.vid_pad)
        land = land.reshape(land.shape[0], -1)
        label = self._load_label(land_p.split('/')[-2])
        anno = self._load_anno(anno_p) # 正解の文字列データのロード

        # trainの場合、水平(垂直)反転
        if(self.phase == 'train'):
            vid = HorizontalFlip(vid)
        # 色の標準化
        vid = ColorNormalize(vid)
        if opt.color_mode == 0:
            vid = vid.reshape(vid.shape[0], vid.shape[1], vid.shape[2], 1)
        
        vid_len = vid.shape[0] # ビデオの数
        anno_len = anno.shape[0] # アノテーションの数

        vid = self._padding(vid, self.vid_pad) # ビデオのパッディング
        anno = self._padding(anno, self.txt_pad) # アノテーションのパッディング
        
        return {
            'vid': torch.FloatTensor(vid.transpose(3, 0, 1, 2)),
            'land': torch.FloatTensor(land),
            'label': torch.LongTensor(label),
            'txt': torch.LongTensor(anno),
            'txt_len': anno_len,
            'vid_len': vid_len}

    # データの長さを返す関数
    def __len__(self):
        return len(self.video_data)
    
    # ビデオのロード
    def _load_vid(self, p):
        files = os.listdir(p) # パスで指定された場所に入っているものの一覧を配列にして返す
        # フィルタリング .pngを見つける
        files = list(filter(lambda file: file.find('.png') != -1, files))# 構文みたいなもの
        files = sorted(files, key=lambda file: int(os.path.splitext(file)[0])) # splitextは拡張子(.jpgなど)を抽出できる
        if self.color_mode == 0:
            array = [cv2.imread(os.path.join(p, file), cv2.IMREAD_GRAYSCALE) for file in files] # フレームデータのロード
        else:
            array = [cv2.imread(os.path.join(p, file)) for file in files] # フレームデータのロード
        array = list(filter(lambda im: not im is None, array)) # データないものをフィルタリングする
        array = np.stack(array, axis=0).astype(np.float32)
        return array

    # アノテーションのロード
    def _load_anno(self, name):
        with open(os.path.join(name), 'r') as f:
            lines = (f.read()).split('\n')
            if '' in lines:
                lines.remove('')
            txt = lines[0].split(' ')
            text = []
            for t in txt:
                if self.txt_kind == 'anno_data_romaji':
                    text.append(t.upper())
                if self.txt_kind == 'anno_data_mydic':
                    text.append(t)
        return MyDataset.txt2arr(' '.join(text), 1)
    
    def _load_land(self, p):
        land = []
        with open(p, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                row.remove('')
                number = list(map(float, row))
                number = np.array(number)
                number = number.reshape(number.shape[0]//2, 2)
                # number[:, 1] *= -1
                if self.land_input_mode > 0:
                    land.append(number)
                elif self.land_input_mode == 0:
                    land.append(np.linalg.norm(number[48:, :]))
        land = np.array(land)

        if self.land_input_mode == 0:
            land_min = land.min()
            land_max = land.max()
            land = (land - land_min) \
                / (land_max - land_min)
        if self.land_input_mode == 1:
            land_min_w = None
        if self.land_input_mode == 2:
            land = [land[i]-land[i+1] for i in range(land.shape[0]-1)]
            land = np.stack(land, axis=0).astype(np.float32)
        
        if self.land_input_mode == 3:
            land[:, :, 0] -= self.land_mean[0]
            land[:, :, 0] /= self.land_std[0]
            land[:, :, 1] -= self.land_mean[1]
            land[:, :, 1] /= self.land_std[1]
        return land

    def _load_label(self, kind):
        result = np.zeros(1,)
        result[0] = MyDataset.labels.index(kind)
        return result
    
    def _land_padding(self, array, length):
        size = length - array.shape[0]
        pd_arr = None
        if self.land_input_mode == 0:
            pd_arr = np.zeros([size,])
        else:
            pd_arr = np.zeros([size, array.shape[1], array.shape[2]])
        result = np.vstack([array, pd_arr]) if self.land_input_mode \
              else np.hstack(np.hstack([array, pd_arr]))
        return result


    # パッディング
    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)


    @staticmethod
    def txt2arr(txt, start):
        arr = []
        for c in list(txt):
            arr.append(MyDataset.letters.index(c) + start)
        return np.array(arr)

    @staticmethod
    def arr2txt(arr, start):
        txt = []
        for n in arr:
            if(n >= start):
                txt.append(MyDataset.letters[n - start])
        return ''.join(txt).strip()

    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if(pre != n and n >= start):
                if(len(txt) > 0 and txt[-1] == ' ' and MyDataset.letters[n - start] == ' '):
                    pass
                else:
                    txt.append(MyDataset.letters[n - start])
            pre = n
        return ''.join(txt).strip()

    @staticmethod
    def wer(predict, truth):
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return wer

    @staticmethod
    def cer(predict, truth):
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return cer

if __name__ == '__main__':
    # import sys
    # sys.setrecursionlimit(2000)
    # def load_test(count , max, dataset):
    #     if max <= count:
    #         return
    #     load_test(count+1, max, dataset)
    #     data = dataset.__getitem__(count)
    #     print(data.get('vid'))
    #     print(data.get('txt'))
    import options.options_lipnet as opt

    dataset = MyDataset(
        'data',
        'anno_data',
        'val_lips_path.txt',
        1120,
        270,
        'val',
        opt.anno_kind,
        opt.color_mode
    )
    print(dataset.__getitem__(0).get('vid').size())