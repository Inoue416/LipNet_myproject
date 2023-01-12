import os
import random

trainf = "emotion_train_path.txt"
valf = "emotion_val_path.txt"
train_data = []
val_data = []
data = None
with open("ita_lips_emotion.txt", "r") as f:
    data = (f.read()).split("\n")
while '' in data:
    data.remove('')

data.sort()
all_data = {}
for d in data:
    elem = d.split('/')
    elem.remove('')
    if 'recitation' in elem:
        continue
    if not (elem[-2] in all_data.keys()):
        all_data[elem[-2]] = [d]
    else:
        all_data[elem[-2]].append(d)

for key in all_data.keys():
    buf = all_data[key]
    random.shuffle(buf)
    if "whis" == key:
        for v in buf[0:10]:
            val_data.append(v)
        for t in buf[10:]:
            train_data.append(t)
    else:
        for v in buf[0:40]:
            val_data.append(v)
        for t in buf[40:]:
            train_data.append(t)

with open(trainf, 'w') as f:
    for td in train_data:
        f.write(td+'\n')

with open(valf,'w') as f:
    for vd in val_data:
        f.write(vd+'\n')
