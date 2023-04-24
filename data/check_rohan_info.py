import os
import matplotlib.pyplot as plt
import random

path_file = './rohan_lips_all_path.txt'


def get_vid_lens(paths):
    result = []
    for path in paths:
        leng = len(os.listdir(path))
        result.append(leng)
    return result


def get_txt_lens(paths):
    result = []
    for path in paths:
        with open(path, 'r') as f:
            txt = f.read()
            result.append(len(txt))
    return result


def plot_hist(data):
    plt.hist(data)
    plt.show()


def write_train_val_path(paths, save_path):
    with open(save_path, 'w') as f:
        for path in paths:
            f.write(path+'\n')


def separate_train_val(paths):
    train_rate = 0.9
    random.shuffle(paths)
    random.shuffle(paths)
    train_paths = paths[:int(train_rate*len(paths))]
    val_paths = paths[int(train_rate*len(paths)):]
    return train_paths, val_paths
    

if __name__ == '__main__':
    vid_paths = []
    txt_romaji_paths = []
    txt_mydic_paths = []
    with open(path_file, 'r') as f:
        for name in f.readlines():
            name = name.rstrip()
            vid_paths.append(name)
            txt_mydic_path = name.replace('/lips', '/anno_data_mydic') + '.txt'
            txt_mydic_paths.append(txt_mydic_path)
            txt_romaji_path = name.replace('/lips', '/anno_data_romaji') + '.txt'
            txt_romaji_paths.append(txt_romaji_path)
    
    vid_lengs = get_vid_lens(vid_paths)
    vid_lengs.sort()
    txt_romaji_lengs = get_txt_lens(txt_romaji_paths)
    txt_romaji_lengs.sort()
    txt_mydic_lengs = get_txt_lens(txt_mydic_paths)
    txt_mydic_lengs.sort()
    
    print('vid max: {}\nvid min: {}\n'.format(
        vid_lengs[-1], vid_lengs[0]
    ))
    print('txt romaji max: {}\ntxt romaji min: {}\n'.format(
        txt_romaji_lengs[-1], txt_romaji_lengs[0]
    ))
    print('txt mydic max: {}\ntxt mydic min: {}\n'.format(
        txt_mydic_lengs[-1], txt_mydic_lengs[0]
    ))

    plot_hist(vid_lengs)

    train_paths, val_paths = separate_train_val(vid_paths)
    write_train_val_path(train_paths, 'rohan4600_train.txt')
    write_train_val_path(val_paths, 'rohan4600_val.txt')