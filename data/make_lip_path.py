import os
import random

kinds = ["rohan", "ita"]

train_data = []
val_data = []

# rohan
filename = "./rohan_lips_all_path.txt"
with open(filename, 'r') as f:
    data = (f.read()).split('\n')
    data.remove('')
    random.shuffle(data)
    val_data = data[:608]
    train_data = data[608:]

# ita
filename = "./ita_lips_path.txt"
human = ['zundamon', 'sikokumetan', 'tohokuitako', 'kyusyusora']
vmax = 228
with open(filename, 'r') as f:
    data = (f.read()).split('\n')
    data.remove('')
    random.shuffle(data)

    for h in human:
        v_n = 0
        for d in data:
            if h in d:
                if vmax > v_n:
                    val_data.append(d)
                    v_n += 1
                else:
                    train_data.append(d)

with open('train_lips_path.txt', 'w') as f:
    for td in train_data:
        f.write(td+'\n')

with open('val_lips_path.txt', 'w') as f:
    for vd in val_data:
        f.write(vd+'\n')