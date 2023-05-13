import os
gpu = '0'
random_seed = 0
data_type = 'unseen'
video_path = 'data/'
save_folder_root = '/media/yuyainoue/neelabHDD/yuyainoueHDD/train_log'
train_list = 'rohan4600_train.txt'
val_list = 'rohan4600_val.txt'
data_len = 'rohan4600_only'
anno_path = 'anno_data'
anno_kind = anno_path+'_romaji'
vid_padding = 459
# rohan only: 459
# vid_pad 289: frame99to290
# vid ita zundamon only: max 639, min 84
# ita not have recitation: max 1112, min 57
txt_padding = 116
# ita zundamon only: romaji: max.271, min.6, mydic: max.268, min.5
# rohan only: romaji: 116, mydic: 118
# txt_pad 137: frame99to290 and romaji  # text padding train max -> 137, train min -> 5, val max -> 53, val min -> 49
batch_size = 4
save_step = 500
base_lr = 0.0001
num_workers = 2
max_epoch = 60
train_display = 50
test_display = 50
test_step = 517
year = 2023
month = 0
day = 12
dropout3d_p = 0.3
decay = 1e-4
train_num = 4140  # TODO: 書き換え
val_num = 460  # TODO: 書き換え
color_mode = 0
is_optimize = True
# weights = 'weights/2023_1_30_t6076v1520_Wdecay0_dp0.30_LipNet_unseen_loss_1.037536859512329_wer_4.254112161946642_cer_4.952706367616731.pt'
# weights = 'weights/2023_4_21_t4140v460_Wdecay1e-05_dp0.5_lipnet_sep2_unseen_loss_2.5460121631622314_wer_1.0168597234469439_cer_1.2041710541739.pt.pt'

# LSTMのオプション
hidden_size = 256
num_layers = 1
bidirectional = True
is_batch_first = False
dropout_p = 0.5
land_input_mode = 3
land_x_mean = 148.234
land_y_mean = 51.538
land_x_std = 77.05
land_y_std = 97.665

connect_kind = 'add'
act = 'ReLU'


model_mode = 'lip_only'
model_name = 'lipnet' #'lipnet_land_'+connect_kind+'_'+act

two_to_three = "fc"

lip_only = True

if model_mode == 'lip_only':
    save_prefix = f'_w_{year}_{month}_{day}_t{train_num}v{val_num}_decay{decay}_dp3d{dropout3d_p}_vp{vid_padding}_tp{txt_padding}_batch{batch_size}_ep{max_epoch}_{anno_kind}_{data_len}_{model_mode}_{model_name}_colorMode{color_mode}_TwoToThree_{two_to_three}/{year}_{month}_{day}_t{train_num}v{val_num}_Wdecay{decay}_dp3d{dropout3d_p}_{model_name}_{data_type}'
    log_dir=f"{base_lr}_l_{year}_{month}_{day}_t{train_num}v{val_num}_decay{decay}_dp3d{dropout3d_p}_vp{vid_padding}_tp{txt_padding}_batch{batch_size}_ep{max_epoch}_{anno_kind}_{data_len}_{model_mode}_{model_name}_colorMode{color_mode}_TwoToThree_{two_to_three}"

else:
    save_prefix = f'_w_{year}_{month}_{day}_t{train_num}v{val_num}_decay{decay}_dp3d{dropout3d_p}_vp{vid_padding}_tp{txt_padding}_batch{batch_size}_ep{max_epoch}_{anno_kind}_{data_len}_{model_mode}_{model_name}_colorMode{color_mode}_his{hidden_size}\
        nl{num_layers}_bid{bidirectional}_bf{is_batch_first}_dp{dropout_p}/{year}_{month}_{day}_t{train_num}v{val_num}_Wdecay{decay}_dp3d{dropout3d_p}_{model_name}_{data_type}'
    log_dir=f"{base_lr}_l_{year}_{month}_{day}_t{train_num}v{val_num}_decay{decay}_dp3d{dropout3d_p}_vp{vid_padding}_tp{txt_padding}_batch{batch_size}_ep{max_epoch}_{anno_kind}_{data_len}_{model_mode}_{model_name}_colorMode{color_mode}_his{hidden_size}\
        nl{num_layers}_bid{bidirectional}_bf{is_batch_first}_dp{dropout_p}"
