import os
from tqdm import tqdm

FILES = [
    'emotion_train_path.txt',
    'emotion_val_path.txt'
]

romaji_root = '/media/yuyainoue/neelabHDD/yuyainoueHDD/ITA_data/anno_data_romaji/emotion'
mydic_root = '/media/yuyainoue/neelabHDD/yuyainoueHDD/ITA_data/anno_data_mydic/emotion'

vid_lengs = []
romaji_lengs = []
mydic_lengs = []
anno_exists = []

for filename in FILES:
    paths = None
    with open(filename, 'r') as f:
        paths = [s.rstrip() for s in f.readlines()]
    for path in tqdm(paths):
        vid_lengs.append(len(os.listdir(path)))
        path_sep = path.split('/')
        
        number = path_sep[-1][-3:]
        if number in anno_exists:
            continue
        txt_filename = 'emotion100_'+number+'.txt'
        romaji_path = os.path.join(romaji_root, txt_filename)
        with open(romaji_path, 'r') as f:
            txt = (f.read()).split('\n')[0]
            romaji_lengs.append(len(txt))
        mydic_path = os.path.join(mydic_root, txt_filename)
        with open(mydic_path, 'r') as f:
            txt = (f.read().split('\n'))[0]
            mydic_lengs.append(len(txt))
        anno_exists.append(number)

print('\nvid max: ', max(vid_lengs))
print('vid min: ', min(vid_lengs))
print('\nromaji_lengs max: ', max(romaji_lengs))
print('mydic_lengs max', max(mydic_lengs))
