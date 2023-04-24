import os
import random
from tqdm import tqdm

ROOT = '/media/yuyainoue/neelabHDD/yuyainoueHDD/ITA_data/zundamon'
DATA = 'lips'
DATA_KINDS = [
    'recitation', 'normal'
]

def write_path(paths, filename):
    with open(filename, 'w') as f:
        print('Writting results ...')
        for path in tqdm(paths):
            f.write(path + '\n')


def pickup():
    result = []
    for data_kind in DATA_KINDS:
        data_path = os.path.join(ROOT, DATA, data_kind)
        for dataname in os.listdir(data_path):
            result.append(os.path.join(data_path, dataname))
    return result

def make_path_file():
    paths = pickup()
    random.shuffle(paths)
    random.shuffle(paths)
    random.shuffle(paths)

    step = (len(paths)*90) // 100
    train_paths = paths[:step]
    val_paths = paths[step:]
    write_path(train_paths, 'train_ita_zundamon_only.txt')
    write_path(val_paths, 'val_ita_zundamon_only.txt')

def get_ita_zundamon_info():
    FILES = ['train_ita_zundamon_only.txt', 'val_ita_zundamon_only.txt']
    vid_lengs = []
    romaji_lengs = []
    mydic_lengs = []
    for filename in FILES:
        with open(filename, 'r') as f:
            print('Search: ', filename)
            for s in tqdm(f.readlines()):
                path = s.rstrip()
                vid_lengs.append(len(os.listdir(path)))
                path_sep = path.split('/')
                romaji_path = None
                mydic_path = None
                txt_root = '/'.join(path_sep[:6])
                romaji_path = os.path.join(txt_root, 'anno_data_romaji')
                mydic_path = os.path.join(txt_root, 'anno_data_mydic')
                if path_sep[-2][0] == 'r':
                    romaji_path = os.path.join(romaji_path, path_sep[-2])
                    mydic_path = os.path.join(mydic_path, path_sep[-2])
                    txt_filename = path_sep[-2] + '324_' + path_sep[-1][-3:] + '.txt'
                    romaji_path = os.path.join(romaji_path, txt_filename)
                    mydic_path = os.path.join(mydic_path, txt_filename)
                else:
                    romaji_path = os.path.join(romaji_path, 'emotion')
                    mydic_path = os.path.join(mydic_path, 'emotion')
                    txt_filename = 'emotion100_' + path_sep[-1][-3:] + '.txt'
                    romaji_path = os.path.join(romaji_path, txt_filename)
                    mydic_path = os.path.join(mydic_path, txt_filename)
                
                with open(romaji_path, 'r') as f:
                    txt = (f.readlines()[0]).rstrip()
                    romaji_lengs.append(len(txt))

                with open(mydic_path, 'r') as f:
                    txt = (f.readlines()[0]).rstrip()
                    mydic_lengs.append(len(txt))
    print('\n' + ('-'*10) + ' Result ' + ('-'*10))
    print('vid max leng: ', max(vid_lengs))
    print('vid min leng: ', min(vid_lengs))
    print('\nromaji max leng: ', max(romaji_lengs))
    print('romaji min leng: ', min(romaji_lengs))
    print('\nmydic max leng: ', max(mydic_lengs))
    print('mydic min leng: ', min(mydic_lengs), '\n')

if __name__ == '__main__':
    get_ita_zundamon_info()