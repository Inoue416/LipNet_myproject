import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import sys
from datasets.dataset_rand import MyDataset
import numpy as np
import time
# from lstm_randlabel import LstmNet
from mymodels.model_detect_lstm_norm import LstmNet
import torch.optim as optim
import re
import json
from tensorboardX import SummaryWriter
import options.options_rand as opt



if(__name__ == '__main__'):
    opt = __import__('options_rand')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    writer = SummaryWriter(log_dir=opt.log_dir) # ダッシュボード作成

# データセットをDataLoaderへ入れてDataLoaderの設定をして返す
def dataset2dataloader(dataset, num_workers=opt.num_workers, shuffle=True):
    return DataLoader(dataset,
        batch_size = opt.batch_size,
        shuffle = shuffle,
        num_workers = num_workers, # マルチタスク
        drop_last = True) # Trueにすることでミニバッチから漏れた仲間外れを除去できる (Trueを検討している)

# 学習率を返す
def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return np.array(lr).mean()

# テスト
@torch.no_grad()
def test(model, net):
    # テストデータのロード
    dataset = MyDataset(
        opt.anno_path,
        opt.val_list,
        opt.vid_padding,
        'test')

    #print('num_test_data:{}'.format(len(dataset.data)))
    model.eval() # テストモードへ
    loader = dataset2dataloader(dataset, shuffle=False) # DataLoaderを作成
    loss_list = []
    crit = nn.CrossEntropyLoss()
    tic = time.time()
    for (i_iter, input) in enumerate(loader):
        rand = input.get('rand').cuda()
        label = input.get('label').cuda()
        label = torch.reshape(label, (opt.batch_size,))

        y = net(rand) # ネットへビデオデータを入れる
        # 損出関数での処理
        loss = crit(y, label).detach().cpu().numpy()
        # 損出関数の値を記録
        loss_list.append(loss)
        
        # 結果の文字を入れる
        # 正しい文字列を入れる

        # 条件の回数の時だけエラー率などを表示
        if(i_iter % opt.display == 0):
            v = 1.0*(time.time()-tic)/(i_iter+1)
            eta = v * (len(loader)-i_iter) / 3600.0

            print(''.join(101*'-'))
            print('{}: {} | {}: {}'.format('predict', torch.argmax(y, dim=1), 'truth', label))
            print(''.join(101*'-'))
            print(''.join(101 *'-'))
            print('test_iter={},loss={}'.format(i_iter,loss))
            print(''.join(101 *'-'))

    return (np.array(loss_list).mean())

# 訓練
def train(model, net):

    # データのロード
    dataset = MyDataset(
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        'train')

    # DataLoaderの作成
    loader = dataset2dataloader(dataset)
    # optimizerの初期化(Adam使用)
    optimizer = optim.Adam(model.parameters(),
                lr = opt.base_lr,
                weight_decay = 0.0001,#.001,#0.01,#0.1, # パラメータのL2ノルムを正規化としてどれくらい用いるから指定
                eps=1e-8,
                amsgrad = True)# AMSgradを使用する
    """optimizer = optim.SGD(model.parameters(),
                lr = opt.base_lr,
                momentum=0.9,
                )"""

    crit = nn.CrossEntropyLoss()
    tic = time.time()
    # TODO:accuracyの準備
    loss_list = []
    for epoch in range(opt.max_epoch): # epoch分学習する
        for (i_iter, input) in enumerate(loader):
            model.train() # 訓練モードへ
            rand = input.get('rand').cuda()
            label = input.get('label').cuda()
            label = torch.reshape(label, (opt.batch_size,))

            optimizer.zero_grad() # パラメータ更新が終わった後の勾配のクリアを行っている。
            y = net(rand) # ビデオデータをネットに投げる
            
            # 損出を求める
            loss = crit(y, label)
            
            loss_list.append(loss)

            # 損出をもとにバックワードで学習
            loss.backward()

            if(opt.is_optimize):
                optimizer.step() # gradプロパティに学習率をかけてパラメータを更新

            tot_iter = i_iter + epoch*len(loader) # 現在のイテレーション数の更新

            # 条件の回数の時、それぞれの経過を表示
            if(tot_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(tot_iter+1)
                eta = (len(loader)-i_iter)*v/3600.0

                writer.add_scalar('train loss', loss, tot_iter)
                print(''.join(101*'-'))
                print('{}: {} | {}: {}'.format('predict', torch.argmax(y, dim=1), 'truth', label))
                print(''.join(101*'-'))
                print(''.join(101*'-'))
                print('epoch={},tot_iter={},base_lr={},eta={},loss_mean={},loss={}'.format(epoch, tot_iter, opt.base_lr, eta, torch.mean(torch.stack(loss_list)), loss))
                print(''.join(101*'-'))

            if(tot_iter % opt.test_step == 0):
                loss = test(model, net)
                print('i_iter={},lr={},loss={}'
                    .format(tot_iter,show_lr(optimizer),loss))
                writer.add_scalar('val loss', loss, tot_iter)
                if (tot_iter % 1000 == 0):
                    savename = 'base_lr{}_{}_losst{}_lossv{}.pt'.format(opt.base_lr, opt.save_prefix, torch.mean(torch.stack(loss_list)), loss) # 学習した重みを保存するための名前を作成
                    (path, name) = os.path.split(savename)
                    if(not os.path.exists(path)): os.makedirs(path) # 重みを保存するフォルダを作成する
                    torch.save(model.state_dict(), savename) # 学習した重みを保存
                if(not opt.is_optimize):
                    exit()

if(__name__ == '__main__'):
    print("Loading options...")
    model = LstmNet() # モデルの定義
    model = model.cuda() # gpu使用
    net = nn.DataParallel(model).cuda() # データの並列処理化

    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(f=opt.weights)# 学習済みの重みをロード
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    # ネットワークの挙動に再現性を持たせるために、シードを固定して重みの初期値を固定できる
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)
    # 訓練開始
    train(model, net)
