import os
import random

FILENAME = './frame_99to290_path.txt'
TRAINFILENAME = './train_frame_99to290_path.txt'
VALFILENAME = './val_frame_99to290_path.txt'
ANNO_KIND = [
    "anno_data_mydic"
    , "anno_data_romaji"
]
DIV = 0.8

def search_txt_lenght(path, kind):
    if 'ROHAN4600_data' in path.split('/'):
        path = path.replace('lips', kind)
        path = path.split('/')
        path[-1] = path[-1][:-1]+'.txt'
    else:
        path = path.replace('lips', kind)
        path = path.split('/')
        path.pop(6)
        if path[7] != 'recitation':
            path[7] = 'emotion'
            num = path[-1][-4:-1]
            path[-1] = 'emotion100_'+num+'.txt'
        else:
            num = path[-1][-4:-1]
            path[-1] = 'recitation324_'+num+'.txt'

    path = '/'.join(path)
    txt = ''
    with open(path, 'r') as f:
        txt = (f.readlines())[0]
    return len(txt)

if __name__ in '__main__':
    train_path = []
    val_path = []
    train_txt_max_len = 0
    train_txt_min_len = 10000
    val_txt_max_len = 0
    val_txt_min_len = 10000
    with open(FILENAME, 'r') as f:
        paths = f.readlines()
        random.shuffle(paths)
        length = len(paths)
        train_num = int(round(length * DIV + 0.1))
        train_path = paths[:train_num]
        val_path = paths[train_num+1:]

    with open(TRAINFILENAME, 'w') as f:
        for tp in train_path:
            for ak in ANNO_KIND:
                length = search_txt_lenght(tp, ak)
                #print(length)
                if train_txt_max_len < length:
                    train_txt_max_len = length
                if train_txt_min_len > length:
                    train_txt_min_len = length
            f.write(tp)

    with open(VALFILENAME, 'w') as f:
        for vp in val_path:
            for ak in ANNO_KIND:
                length = search_txt_lenght(tp, ak)
                #print(length)
                if val_txt_max_len < length:
                    val_txt_max_len = length
                if val_txt_min_len > length:
                    val_txt_min_len = length
            f.write(vp)

    print("train num: {}".format(len(train_path)))
    print("val num: {}".format(len(val_path)))
    print()
    print("train txt max length: {}".format(train_txt_max_len))
    print("train txt min length: {}".format(train_txt_min_len))
    print()
    print("val txt max length: {}".format(val_txt_max_len))
    print("val txt min length: {}".format(val_txt_min_len))