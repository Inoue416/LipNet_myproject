import os
import shutil


# reading folder only
log_dir = 'train_logs'
folders = os.listdir('./')
remove_folders = [
    'test_space',
    'data'
]
buf = []
for folder in folders:
    if '.py' in folder:
        continue
    if '_' == folder[0]:
        continue
    if '.' == folder[0]:
        continue
    if folder in remove_folders:
        continue
    if folder == log_dir:
        continue
    buf.append(folder)
folders = buf

# moving folders
for folder in folders:
    buf = folder.split('_')
    count = 0
    flag = False
    dates = []
    for b in buf:
        if count == 3:
            break
        if flag:
            dates.append(b)
            count += 1
            continue
        if (flag == False) and ('2022' in b or '2023' in b):
            flag = True
            if b[0] != '2':
                b = b[1:]
            dates.append(b)
            count += 1
    refolder = '_'.join(dates)
    target_path = os.path.join(log_dir, refolder)
    try:
        shutil.move(folder, target_path)
    except Exception as e:
        print(e.message)
        exit()
    
    