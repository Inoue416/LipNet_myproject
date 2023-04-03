import numpy as np
import csv
import matplotlib.pyplot as plt


filepath = '/media/yuyainoue/neelabHDD/yuyainoueHDD/ITA_data/zundamon/rand_label/ama/emoAma005.csv'
data = []
origin = []
scala = []
with open(filepath, 'r') as csvfile:
    reader = csv.reader(csvfile)
    count = 0
    for row in reader:
        row.remove('')
        number = list(map(float, row))
        number = np.array(number)
        number = number.reshape(number.shape[0]//2, 2)
        number[:, 1] *= -1
        print(number.shape)
        data.append(number)
        origin.append(number)
        scala.append(np.linalg.norm(number[48:, :]))
        count += 1
    data = np.array(data)
    origin = np.array(origin)
    data = [data[i]-data[i+1] for i in range(data.shape[0]-1)]
    data = np.stack(data, axis=0).astype(np.float32)
    scala = np.array(scala)
# data[:, 48:, :]
data = data[:, 48:, :]
origin = origin[:, 48:, :]
target_o = origin[0]
target_d = data[0]
print(data.shape)
print(origin.shape)
# Max-Min Normalization
max_scala = np.max(scala)
min_scala = np.min(scala)
scala = (scala - min_scala) \
    / (max_scala - min_scala)

print(target_o.shape)
print(target_d.shape)

fig, ax = plt.subplots(1, 1)
# print(ax.shape)
# exit()
x_o = target_o[:, 0].reshape(target_o.shape[0],)
y_o = target_o[:, 1].reshape(target_o.shape[0],)
# ax[0].scatter(x_o, y_o)
# ax[0].set_title('origin randmark')
# ax[0].set_xlabel('x')
# ax[0].set_ylabel('y')
x_d = target_d[:, 0].reshape(target_d.shape[0],)
y_d = target_d[:, 1].reshape(target_d.shape[0],)
x = np.array([i for i in range(scala.shape[0])])
ax.scatter(x=x, y=scala)# x_d, y_d)
ax.set_title('diff randmark')
ax.set_xlabel('x')
ax.set_ylabel('y')
# plt.subplots_adjust(hspace=0.5)
plt.savefig("./zundamon_emoAma005.png")
fig.show()
plt.pause(1000)
