from functools import total_ordering
import os
import matplotlib.pyplot as plt

ROOT = "/media/yuyainoue/neelabHDD/yuyainoueHDD"
DATA_KIND = ['ITA_data', 'ROHAN4600_data']
HUMANS = [
    'kyusyusora',
    'sikokumetan',
    'tohokuitako',
    'zundamon'
]

FRAME_NAME = "lips"


def judge_range(data):
    if  99 < data and data < 290:
        return True
    return False

if __name__ == "__main__":
    result = []
    paths = []
    total_result = []
    max_length = 0
    for dk in DATA_KIND:
        if dk[0] == 'R':
            data_path = os.path.join(ROOT, dk, FRAME_NAME)
            for data in os.listdir(data_path):
                length = len(os.listdir(os.path.join(data_path, data)))
                total_result.append(length)
                if judge_range(length):
                    result.append(length)
                    if max_length < length:
                        max_length = length
                    paths.append(os.path.join(data_path, data))
        else:
            for hk in HUMANS:
                data_path = os.path.join(ROOT, dk, hk, FRAME_NAME)
                for emotion in os.listdir(data_path):
                    emotion_path = os.path.join(data_path, emotion)
                    for d in os.listdir(emotion_path):
                        length = len(os.listdir(os.path.join(emotion_path, d)))
                        total_result.append(length)
                        if judge_range(length):
                            result.append(length)
                            if max_length < length:
                                max_length = length
                            paths.append(os.path.join(emotion_path, d))
    result.sort()
    total_result.sort()
    print(max_length)  # 289
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.hist(total_result, bins=100)
    plt.savefig("./total_hist.png")
    with open('./frame_99to290_path.txt', 'w') as f:
        for path in paths:
            f.write(path+'\n')