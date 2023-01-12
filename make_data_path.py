import os
ita_root = "/media/yuyainoue/neelabHDD/yuyainoueHDD/ITA_data"
rohan_root = "/media/yuyainoue/neelabHDD/yuyainoueHDD/ROHAN4600_data"

target_folders = [
    "zundamon",
    "kyusyusora", 
    "tohokuitako", 
    "sikokumetan"
]
target_kinds = [
    "frames",
    "lips"
]

ita_lips_pf = open('ita_lips_path.txt', 'a')
ita_lips_emo_pf = open('ita_lips_emotion.txt', 'a')
ita_lips_rec_pf = open('ita_lips_recitation.txt', 'a')

ita_frames_pf = open('ita_frames_path.txt', 'a')
ita_frames_emo_pf = open('ita_frames_emotion.txt', 'a')
ita_frames_rec_pf = open('ita_frames_recitation.txt', 'a')


for tf in target_folders:
    path1 = os.path.join(ita_root, tf)
    for tk in target_kinds:
        path2 = os.path.join(path1, tk)
        for k in os.listdir(path2):
            path3 = os.path.join(path2, k)
            for data in os.listdir(path3):
                if tk == 'lips':
                    ita_lips_pf.write(os.path.join(path3, data)+'\n')
                    if k == 'recitation':
                        ita_lips_rec_pf.write(os.path.join(path3, data)+'\n')
                    else:
                        ita_lips_emo_pf.write(os.path.join(path3, data)+'\n')
                else:
                    ita_frames_pf.write(os.path.join(path3, data)+'\n')
                    if k == 'recitation':
                        ita_frames_rec_pf.write(os.path.join(path3, data)+'\n')
                    else:
                        ita_frames_emo_pf.write(os.path.join(path3, data)+'\n')

ita_lips_pf.close()
ita_lips_emo_pf.close()
ita_lips_rec_pf.close()

ita_frames_pf.close()
ita_frames_emo_pf.close()
ita_frames_rec_pf.close()
