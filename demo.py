import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import sys
from dataset import MyDataset
import numpy as np
import time
from model import LipNet
import torch.optim as optim
import re
import json
import tempfile
import shutil
import cv2
import face_alignment
from cvtransforms import ColorNormalize

def cal_area(anno):
    return (anno[:,0].max() - anno[:,0].min()) * (anno[:,1].max() - anno[:,1].min()) 

def output_video(target, txt):
    target = target.split('/')[-1]
    files = os.listdir(os.path.join('data', 'test_frame', target))
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

    font = cv2.FONT_HERSHEY_SIMPLEX
    for file, line in zip(files, txt):
        img = cv2.imread(os.path.join('data', 'test_frame', target, file))
        h, w, _ = img.shape
        img = cv2.putText(img, line, (w//8, 11*h//12), font, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
        img = cv2.putText(img, line, (w//8, 11*h//12), font, 1.0, (255, 255, 255), 0, cv2.LINE_AA)  
        h = h // 2
        w = w // 2
        img = cv2.resize(img, (w, h))     
        cv2.imwrite(os.path.join('data', 'test_frame',target, file), img)
    
    cmd = "ffmpeg -y -i {}/%03d.jpg -r 30 {}".format(os.path.join('data', 'test_frame', target), os.path.join('data', 'output_vid', 'test_'+target+'.MOV'))
    os.system(cmd)


def load_target():
    targets = []
    folders = os.listdir(os.path.join('data', 'test_data'))
    targets = [os.path.join('data', 'test_data', folder) for folder in folders]
    return targets

def load_vid(path):
    files = os.listdir(path)
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
        
    array = [cv2.imread(os.path.join(path, file)) for file in files]
    
    
    array = list(filter(lambda im: not im is None, array))
    video = np.stack(array, axis=0).astype(np.float32)
    video = ColorNormalize(video)
    video = torch.FloatTensor(video.transpose(3, 0, 1, 2))
    return video


def ctc_decode(y):
    y = y.argmax(-1)
    t = y.size(0)
    result = []
    for i in range(t+1):
        result.append(MyDataset.ctc_arr2txt(y[:i], start=1))
    return result
        

if(__name__ == '__main__'):
    opt = __import__('options')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu    
    
    model = LipNet()
    model = model.cuda()
    net = nn.DataParallel(model).cuda()

    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    target_path = load_target()
    for target in target_path:
        video = load_vid(target)
        y = model(video[None,...].cuda())
        txt = ctc_decode(y[0])
        kind = target.split('/')[-1]
        with open('kind_'+kind+'.txt', 'w') as f:
            for t in txt:
                f.write(t+'\n')
        #txt = ''
        output_video(target, txt)