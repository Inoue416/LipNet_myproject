import os
from datasets.dataset import MyDataset
gpu = '0'
random_seed = 0
data_type = 'unseen'
data_root = 'data'
save_folder_root = 'D:/train_log'
train_list = 'train_path.txt'
val_list = 'val_path.txt'
data_len = 'notHaveWhis'
rand_x_mean = 148.234
rand_y_mean = 51.538
rand_x_std = 77.05
rand_y_std = 97.665
vid_padding = 1112
batch_size = 32
save_step = 35
base_lr = 0.0001
num_workers = 2
max_epoch = 300
train_display = 10
test_display = 10
test_step = 35
year = 2023
month = 4
day = 21
input_mode = 3
num_inputs = 136
num_outputs = len(MyDataset.labels)
num_channels = [32, 64, 128, 256]
nc_txt = "_".join([str(i) for i in num_channels])
frame_leng = vid_padding
dropout_p = 0.8
decay = 1e-3
train_num = 2316  # TODO: 書き換え
val_num = 580  # TODO: 書き換え
model_mode = 'rand-xy-AllxyZcoreNormalize'
model_name = 'rand-ditection-lstm'
# save_prefix = os.path.join(
#     f'_w_{year}_{month}_{day}_t{train_num}v{val_num}_decay{decay}_dp{dropout_p}_vp{vid_padding}_batch{batch_size}_ep{max_epoch}_{data_len}_{model_mode}_{model_name}_numChannels{nc_txt}',
#     f'{year}_{month}_{day}_{model_name}_{data_type}')
is_optimize = True
# weights = 'weights/2023_1_30_t6076v1520_Wdecay0_dp0.30_LipNet_unseen_loss_1.037536859512329_wer_4.254112161946642_cer_4.952706367616731.pt'
# weights = 'weights/2023_4_3_t5384v1345_Wdecay0_dp0.30_LipNet+rand_unseen_loss_1.3148791790008545_wer_0.8593508852391988_cer_1.5499585565764304.pt'
# weights = 'weights/1.1209722757339478_1.1209722757339478.pt'
# log_dir=f"{base_lr}_l_{year}_{month}_{day}_t{train_num}v{val_num}_decay{decay}_dp{dropout_p}_vp{vid_padding}_batch{batch_size}_ep{max_epoch}_{data_len}_{model_mode}_{model_name}_numChannels{nc_txt}"

# MyLSTM options
num_layers = 1
bidirectional = True
hidden_size = 16
is_batch_first = False
is_dropout = True

save_prefix = os.path.join(
    f'_w_{year}_{month}_{day}_t{train_num}v{val_num}_decay{decay}_dp{dropout_p}_vp{vid_padding}_batch{batch_size}_ep{max_epoch}_{data_len}'
    ,f'hs{hidden_size}_nl{num_layers}_bid{bidirectional}_bf_{is_batch_first}'
    ,f'{year}_{month}_{day}_{model_name}_{data_type}')

log_dir = os.path.join(
    f"{base_lr}_l_{year}_{month}_{day}_t{train_num}v{val_num}_decay{decay}_dp{dropout_p}_vp{vid_padding}_batch{batch_size}_ep{max_epoch}_{data_len}",
    f"hs{hidden_size}_nl{num_layers}_bid{bidirectional}_bf_{is_batch_first}")
