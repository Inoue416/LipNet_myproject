import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from dataset_lipnet import MyDataset
import numpy as np
from model_lipnet import LipNet
import options_lipnet as opt
import matplotlib.pyplot as plt
import cv2
from cvtransforms import *
import torch.nn.functional as F
from copy import deepcopy
import imageio
from PIL import Image


DATA_FILES = [
    'data/train_lips_path.txt',
    'data/val_lips_path.txt'
]

GRAD_CAM_SAVE_ROOT = '/media/yuyainoue/neelabHDD/yuyainoueHDD/grad_cam_logfig'

ANNO_KIND = 'anno_data_mydic'

WEIGHT = 'weights/2023_1_30_t6076v1520_Wdecay0_dp0.30_LipNet_unseen_loss_1.037536859512329_wer_4.254112161946642_cer_4.952706367616731.pt'

LETTERS = None
if ANNO_KIND == 'anno_data_romaji':
    LETTERS = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
elif ANNO_KIND == 'anno_data_mydic':
    LETTERS = ['a', 'i', 'u', 'e', 'o', 'k', 's', 't', 'n', 'h', 'm', 'y', 'r', 'w', '*', 'g', 'z', 'd', 'b', 'p', 'l', 'v', '^', '-', ' ']

def load_vid(p):
    files = os.listdir(p) # パスで指定された場所に入っているものの一覧を配列にして返す
    # フィルタリング .pngを見つける
    files = list(filter(lambda file: file.find('.png') != -1, files))# 構文みたいなもの
    files = sorted(files, key=lambda file: int(os.path.splitext(file)[0])) # splitextは拡張子(.jpgなど)を抽出できる
    array = [cv2.imread(os.path.join(p, file)) for file in files] # フレームデータのロード
    array = list(filter(lambda im: not im is None, array)) # データないものをフィルタリングする
    array = np.stack(array, axis=0).astype(np.float32)
    return array

def txt2arr(txt, start):
    arr = []
    for c in list(txt):
        arr.append(LETTERS.index(c) + start)
    return np.array(arr)

def load_anno(name):
    with open(os.path.join(name), 'r') as f:
        lines = (f.read()).split('\n')
        if '' in lines:
            lines.remove('')
        txt = lines[0].split(' ')
        text = []
        for t in txt:
            if ANNO_KIND == 'anno_data_romaji':
                text.append(t.upper())
            elif ANNO_KIND == 'anno_data_mydic':
                text.append(t)
    return txt2arr(' '.join(text), 1)


def make_anoo_path(path):
    txt_path = ''
    path_separate = path.split('/')
    path_separate.remove('')
    data_kind = path_separate[4]
    path_separate.remove('lips')
    if data_kind[0] == 'I':  # ITA LOAD
        path_separate[5] = ANNO_KIND
        if path_separate[6][0] != 'r':
            path_separate[6] = 'emotion'
            emotion_number = path_separate[7][-3:]
            path_separate[7] = 'emotion' + '100_' + emotion_number
        else:
            recitatin_number = path_separate[-1][-3:]
            path_separate[-1] = path_separate[-1][:-3] + '324_' + recitatin_number           
    else:
        path_separate.insert(5, ANNO_KIND)
    path_separate[-1] += '.txt'
    txt_path = '/'.join(path_separate)
    txt_path = '/' + txt_path 
    return txt_path


def load_data(data_path):
    # Load and Normalize
    data = load_vid(data_path)
    img1 = deepcopy(data)
    data = ColorNormalize(data)
    data = torch.FloatTensor(data.transpose(3, 0, 1, 2))
    vid_len = torch.tensor([data.shape[1]])
    vid_len = vid_len.unsqueeze(0)
    txt_path = make_anoo_path(data_path)
    txt = load_anno(txt_path)
    anno_len = torch.tensor([txt.shape[0]])
    anno_len = anno_len.unsqueeze(0)
    txt = torch.LongTensor(txt)
    return data, vid_len, txt, anno_len, img1


def predict(model, data, txt, vid_len, txt_len):
    model = model.eval()
    data = data.unsqueeze(0)
    out, feature = model(data)
    crit = nn.CTCLoss()
    loss = crit(out.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1))
    loss.backward()
    feature_vec = feature.grad
    feature_vec = feature_vec.view(
        feature_vec.size(0), feature_vec.size(1),
        feature_vec.size(2), feature_vec.size(3)*feature_vec.size(4))
    return feature, feature_vec

def toHeatmap(x):
    x = (x*255).reshape([x.shape[0], -1])
    cm = plt.get_cmap('jet')
    buf = []
    for xi in x[:,]:
        buf.append(np.array([cm(int(np.round(a)))[:3] for a in xi]))
    x = np.array(buf)
    return x.reshape([x.shape[0], 128, 64, 3])

def make_grad_save_path(data_path, phase, is_check=False):
    path_separate = data_path.split('/')
    data_kind = path_separate[5]
    remake_path = None
    if data_kind[0] == 'R':
        data_number = path_separate[-1]
        remake_path = os.path.join(GRAD_CAM_SAVE_ROOT, phase, data_kind, data_number)
    elif data_kind[0] == 'I':
        data_number = path_separate[-1]
        human_name = path_separate[6]
        emotion_or_recitation = path_separate[-2]
        remake_path = os.path.join(GRAD_CAM_SAVE_ROOT, phase, data_kind, 
                                   human_name, emotion_or_recitation, data_number)
    print(remake_path)
    if is_check:
        return remake_path
    if not os.path.exists(remake_path):
        os.makedirs(remake_path)

    return remake_path


def make_gif(data_path, is_check=False):
    save_path = '/'.join(data_path.split('/')[:-1])
    outname = data_path.split('/')[-1]
    data = []
    if is_check:
        return os.path.join(save_path, outname + '.gif')
    files = os.listdir(data_path)
    files.sort()
    for file in files:
        img = Image.open(os.path.join(data_path, file))
        data.append(img)
    data[0].save(os.path.join(save_path, outname + '.gif'),
                 save_all=True, append_images=data[1:], optimize=False, duration=0, loop=0)


def check_data(data_path, phase):
    count = 0
    grad_frame_path = make_grad_save_path(data_path, phase, True)
    grad_gif_path = make_gif(grad_frame_path, True)
    if os.path.exists(grad_frame_path):
        if len(os.listdir(data_path)) == len(os.listdir(grad_frame_path)):
            count += 1
    if os.path.exists(grad_gif_path):
        count += 1
    return count


def grad_cam(data_path, model, phase):
    skip_count = check_data(data_path, phase)
    grad_save_path = None
    if skip_count == 2:
        return
    if skip_count == 0:
        data, vid_len, txt, anno_len, img1 = load_data(data_path)
        feature, feature_vec = predict(model, data, txt, vid_len, anno_len)

        feature_vec = feature_vec.squeeze(0)
        # alphaパラメータを作る
        alpha = torch.mean(feature_vec, dim=2)

        alpha = alpha.view(alpha.size(0), alpha.size(1), 1, 1)
        feature_map = feature.squeeze(0)

        L = F.relu(torch.sum(feature_map*alpha, dim=0))
        L = L.detach().numpy()
        # 0-1で正規化
        # Min-Max Normalization
        L_min = np.min(L)
        L_max = np.max(L) - L_min
        L = (L - L_min) / L_max
        L = L.reshape([L.shape[2], L.shape[1], L.shape[0]])
        # print(L.shape)
        # もとのサイズにリサイズ
        # OpenCVに使用上チャネルの次元数が512以下でないといけない
        # そのため512より大きいものは3つに分解してリサイズ後くっつける
        if L.shape[-1] <= 512:
            L = cv2.resize(L, (128, 64))
            L = L.reshape([L.shape[2], L.shape[1], L.shape[0]])
        else:
            sep_num = L.shape[-1] // 3
            sepL1 = cv2.resize(deepcopy(L[:, :, 0:sep_num]), (128, 64))
            sepL1 = sepL1.reshape([sepL1.shape[2], sepL1.shape[1], sepL1.shape[0]])
            sepL2 = cv2.resize(deepcopy(L[:, :, sep_num:2*sep_num]), (128, 64))
            sepL2 = sepL2.reshape([sepL2.shape[2], sepL2.shape[1], sepL2.shape[0]])
            sepL3 = cv2.resize(deepcopy(L[:, :, 2*sep_num:]), (128, 64))
            sepL3 = sepL3.reshape([sepL3.shape[2], sepL3.shape[1], sepL3.shape[0]])
            L = np.vstack((sepL1, sepL2, sepL3))
        img1 = img1[:, :, :, ::-1] # permute(1, 3, 2, 0)
        img1 = img1 / 255
        # img1 = img1[0]
        img2 = toHeatmap(L)
        img2 = img2.reshape([img2.shape[0], img2.shape[2], img2.shape[1], img2.shape[3]])
        alpha = 0.4
        grad_cam_img = img1*alpha + img2*(1-alpha)
        grad_cam_img = 255 * (grad_cam_img / np.max(grad_cam_img))
        grad_cam_img = grad_cam_img.astype(np.uint8)
        
        grad_save_path = make_grad_save_path(data_path, phase)
        print("Process make frames: {}".format(grad_save_path))
        for idx in range(grad_cam_img.shape[0]):
            filename = str(idx).zfill(3) + '.png'
            # print(filename)
            buf2 = grad_cam_img[idx]
            buf2 = buf2[:, :, :]
            imageio.imwrite(os.path.join(grad_save_path, filename), buf2)
        print('fin\n')
    else:
        grad_save_path = make_grad_save_path(data_path, phase)

    print('Process make gif: {}'.format(grad_save_path))
    make_gif(grad_save_path)
    print('fin\n')


if __name__ == '__main__':
    train_path = []
    val_path = []
    if not os.path.exists(GRAD_CAM_SAVE_ROOT):
        os.makedirs(GRAD_CAM_SAVE_ROOT)

    model = LipNet(color_mode=1, dictype='anno_data_mydic', is_grad=True) # モデルの定義
    pretrained_dict = torch.load(f=WEIGHT)# 学習済みの重みをロード
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
    print('miss matched params:{}'.format(missed_params))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    with open(DATA_FILES[0], 'r') as f:
        train_path = [s.rstrip() for s in f.readlines()]

    with open(DATA_FILES[1], 'r') as f:
        val_path = [s.rstrip() for s in f.readlines()]
    
    # phase is train.
    for tp in train_path:
        grad_cam(tp, model, 'train')
    
    for vp in val_path:
        grad_cam(vp, model, 'val')
    
    print('\nAll Process Complete.')

