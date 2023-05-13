import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from datasets.dataset_lipnet_land_multi import MyDataset
import numpy as np
import time
from mymodels.model_lipnet_land_multi import LipNetLandMulti
import torch.optim as optim
from tensorboardX import SummaryWriter
import options.options_lipnet_land_multi as opt



if(__name__ == '__main__'):
    # opt = __import__('options.options_lipnet_land_multi')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    logdir = os.path.join(opt.save_folder_root, 
                        '_'.join([str(s) for s in [opt.year, opt.month, opt.day]]), opt.log_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir) # ダッシュボード作成

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

# CTC損出関数での結果を文字へと変換する
def ctc_decode(y):
    result = []
    y = y.argmax(-1) 
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]

def calcurate_detect_accuracy(out, target):
    result = out == target
    total = out.shape[0]
    return np.sum(result) * 100 / total

# テスト
def test(model, net):

    with torch.no_grad(): # テストなどで勾配は求めない処理
        # テストデータのロード
        dataset = MyDataset(opt.video_path,
            opt.anno_path,
            opt.val_list,
            opt.vid_padding,
            opt.txt_padding,
            'test', 
            opt.anno_kind,
            opt.color_mode,
            land_mean=[opt.land_x_mean, opt.land_y_mean],
            land_std=[opt.land_x_std, opt.land_y_std],
            land_input_mode=opt.land_input_mode)

        #print('num_test_data:{}'.format(len(dataset.data)))
        model.eval() # テストモードへ
        loader = dataset2dataloader(dataset, shuffle=False) # DataLoaderを作成
        loss_list = []
        wer = []
        cer = []
        acc_labels = []
        crit = nn.CTCLoss()
        cross = nn.CrossEntropyLoss()
        tic = time.time()
        for (i_iter, input) in enumerate(loader):
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            land = input.get('land').cuda()
            label = input.get('label').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()
            #print(vid_len)
            #print(txt_len)
            y, y2 = net(vid, land) # ネットへビデオデータを入れる
            pred_label = torch.argmax(y2.clone().detach(), dim=1).cpu().numpy()
            truth_label = label.clone().detach().cpu().numpy()
            truth_label = truth_label.reshape(-1)
            acc_label = calcurate_detect_accuracy(pred_label, truth_label)
            # 損出関数での処理
            loss1 = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1)).detach().cpu().numpy()
            label = torch.reshape(label, (opt.batch_size,))
            loss2 = cross(y2, label).detach().cpu().numpy()
            loss = loss1 + loss2
            #print(loss)
            # 損出関数の値を記録
            loss_list.append(loss)
            # 結果の文字を入れる
            pred_txt = ctc_decode(y)
            # 正しい文字列を入れる
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            # 単語及び、文字のエラー率を算出
            wer.extend(MyDataset.wer(pred_txt, truth_txt))
            cer.extend(MyDataset.cer(pred_txt, truth_txt))
            acc_labels.append(acc_label)

            # 条件の回数の時だけエラー率などを表示
            if(i_iter % opt.test_display == 0):
                print(''.join(101*'-'))
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-'))
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:4]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101 *'-'))
                print('Truth label: ', truth_label)
                print('Predict label: ', pred_label)
                print(''.join(101 *'-'))
                print('Label Accuracy: {:.2f}%'.format(acc_label))
                print(''.join(101 *'-'))
                print('test_iter={},loss={},wer={},cer={},acc_label={}'.format(i_iter,loss,np.array(wer).mean(),np.array(cer).mean(),np.array(acc_labels).mean()))
                print(''.join(101 *'-'))
        
        print('\n\ntest finish time: {:.2f} s\n\n'.format(time.time() - tic))
        return (np.array(loss_list).mean(), np.array(wer).mean(), np.array(cer).mean(), np.array(acc_labels).mean())

# 訓練
def train(model, net):
    # データのロード
    dataset = MyDataset(
        opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'train',
        opt.anno_kind,
        opt.color_mode,
        land_mean=[opt.land_x_mean, opt.land_y_mean],
        land_std=[opt.land_x_std, opt.land_y_std],
        land_input_mode=opt.land_input_mode)

    # DataLoaderの作成
    loader = dataset2dataloader(dataset)
    # optimizerの初期化(Adam使用)
    optimizer = optim.Adam(model.parameters(),
                lr = opt.base_lr,
                weight_decay = opt.decay,   # パラメータのL2ノルムを正規化としてどれくらい用いるから指定
                eps=1e-8,
                amsgrad = True)# AMSgradを使用する"""
    """optimizer = optim.SGD(model.parameters(),
                lr = opt.base_lr,
                momentum=0.9,
                )"""

    #print('num_train_data:{}'.format(len(dataset.data)))
    crit = nn.CTCLoss()
    cross = nn.CrossEntropyLoss()
    tic = time.time()
    train_wer = []
    train_cer = []
    loss_list = []
    acc_labels = []
    for epoch in range(opt.max_epoch): # epoch分学習する
        epoch_tic = time.time()
        for (i_iter, input) in enumerate(loader):
            model.train() # 訓練モードへ
            vid = input.get('vid').cuda()
            land = input.get('land').cuda()
            label = input.get('label').cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()
            #if vid_len.view(-1) < txt_len.view(-1):
            #print('vl : {}'.format(vid_len.view(-1)))
            #print('tl : {}'.format(txt_len.view(-1)))

            if not opt.is_batch_first:
                land = land.view(land.size(1), land.size(0), land.size(2)).contiguous()

            optimizer.zero_grad() # パラメータ更新が終わった後の勾配のクリアを行っている。
            y, y1 = net(vid, land) # ビデオデータをネットに投げる
            # 損出を求める
            pred_label = torch.argmax(y1.clone().detach(), dim=1).cpu().numpy()
            truth_label = label.clone().detach().cpu().numpy()
            truth_label = truth_label.reshape(-1)
            acc_label = calcurate_detect_accuracy(pred_label, truth_label)
            acc_labels.append(acc_label)
            loss1 = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1))
            label = torch.reshape(label, (opt.batch_size,))            
            loss2 = cross(y1, label)
            loss = loss1 + loss2
            loss_list.append(loss)
            #print(loss)
            # 損出をもとにバックワードで学習
            loss.backward()
            if(opt.is_optimize):
                optimizer.step() # gradプロパティに学習率をかけてパラメータを更新

            tot_iter = i_iter + epoch*len(loader) # 現在のイテレーション数の更新
            pred_txt = ctc_decode(y) # 結果を文字へ
            # 正解の文字列をロード
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
        
            #exit()
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt)) # エラー率を算出
            train_cer.extend(MyDataset.cer(pred_txt, truth_txt))

            # 条件の回数の時、それぞれの経過を表示
            if(tot_iter % opt.train_display == 0):
                writer.add_scalar('train loss ctc', loss1, tot_iter)
                writer.add_scalar('train wer', np.array(train_wer).mean(), tot_iter)
                writer.add_scalar('train cer', np.array(train_cer).mean(), tot_iter)
                writer.add_scalar('train loss cross', loss2, tot_iter)
                writer.add_scalar('train label accuracy', acc_label, tot_iter)
                print(''.join(101*'-'))
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-'))
                print('Truth label: ', truth_label)
                print('Predict label: ', pred_label)
                print(''.join(101*'-'))
                print('Label Accuracy: {:.2f}%'.format(acc_label))
                print(''.join(101*'-'))

                for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101*'-'))
                print('epoch={},tot_iter={},base_lr={},loss_mean={}loss={},train_wer={},train_cer={},train_acc_label={:.2f}'.format(epoch, tot_iter, opt.base_lr, torch.mean(torch.stack(loss_list)), loss, np.array(train_wer).mean(), np.array(train_cer).mean(), acc_label))
                print(''.join(101*'-'))

            if(tot_iter % opt.test_step == 0):
                (loss, wer, cer, acc_label) = test(model, net)
                print('i_iter={},lr={},loss={},wer={},cer={},acc_label={:.2f}'
                    .format(tot_iter,show_lr(optimizer),loss,wer,cer,acc_label))
                writer.add_scalar('val loss', loss, tot_iter)
                writer.add_scalar('wer', wer, tot_iter)
                writer.add_scalar('cer', cer, tot_iter)
                writer.add_scalar('acc label', acc_label, tot_iter)

            if (tot_iter % opt.save_step == 0):
                if not os.path.exists(os.path.join(opt.save_folder_root, '_'.join([str(s) for s in [opt.year, opt.month, opt.day]]))):
                    os.makedirs(os.path.join(opt.save_folder_root, '_'.join([str(s) for s in [opt.year, opt.month, opt.day]])))
                savename = 'base_lr{}{}_loss_{}_wer_{}_cer_{}_accLabel_{}.pt'.format(opt.base_lr, opt.save_prefix, torch.mean(torch.stack(loss_list)),  np.array(train_wer).mean(), np.array(train_cer).mean(), np.array(acc_labels).mean()) # 学習した重みを保存するための名前を作成
                # print(savename)
                (path, name) = os.path.split(savename)
                # print(path)
                # print(name)
                
                if(not os.path.exists(os.path.join(opt.save_folder_root, '_'.join([str(s) for s in [opt.year, opt.month, opt.day]]), path))):
                    os.makedirs(os.path.join(opt.save_folder_root, '_'.join([str(s) for s in [opt.year, opt.month, opt.day]]), path)) # 重みを保存するフォルダを作成する
                torch.save(model.state_dict(), 
                            os.path.join(opt.save_folder_root, '_'.join([str(s) for s in [opt.year, opt.month, opt.day]]), path, name)) # 学習した重みを保存
                
            if(not opt.is_optimize):
                exit()

        epoch_end_tic = time.time() - epoch_tic
        print('\n\nepoch{} finish time: {:.2f} s\n\n'.format(epoch, epoch_end_tic))
    
    print('\n\nAll process finish.')
    print('time: {:.2f} s'.format(time.time() - tic))

if(__name__ == '__main__'):
    print("Loading options...")
    model = LipNetLandMulti(T=opt.vid_padding, hidden_size=opt.hidden_size, num_layers=opt.num_layers,
                            color_mode=opt.color_mode, anno_kind=opt.anno_kind, bidirectional=opt.bidirectional,
                            batch_first=opt.is_batch_first, dropout_p=opt.dropout_p, dropout3d_p=opt.dropout3d_p) # モデルの定義
    
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
