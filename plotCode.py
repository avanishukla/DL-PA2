import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

dd = np.array(re.findall(r"[^,\s\n]+", open("log_train1280001bn.txt", "r").read())).reshape(-1,6)
epoch = dd[:,1].astype(int)
steps = (dd[:,3].astype(float)).astype(int)
loss = dd[:,5].astype(float)
updation_per_epoch = 15
# sr_epoch = 3.7
# total_epoch =5
steps = steps[0:updation_per_epoch]
loss = loss[0:updation_per_epoch]
fig = plt.figure()
p1, = plt.plot(steps, loss,'b',label='training loss')
# plt.grid()
# plt.xticks([w*updation_per_epoch for w in range(total_epoch)], ['%i'%w for w in range(total_epoch)])
fig.suptitle('Loss across parameter updates', fontsize=15)
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('negative log likelihood', fontsize=16)
#plt.axvspan(sr_epoch*updation_per_epoch, total_epoch*updation_per_epoch, facecolor='r', alpha=0.2,zorder=-1000)
#extra = Rectangle((0, 0), 1, 1, fc="#F08080", fill=True, edgecolor='#F08080', linewidth=1)
#plt.legend([p1,extra], ["Training Loss","Overfitting Region"],loc='upper right')
#plt.legend([p1], ["Training Loss"],loc='upper right')
#plt.show()

dd = np.array(re.findall(r"[^,\s\n]+", open("log_val1280001bn.txt", "r").read())).reshape(-1,6)
epoch = dd[:,1].astype(int)
steps = (dd[:,3].astype(float)).astype(int)
loss = dd[:,5].astype(float)
steps = steps[0:updation_per_epoch]
loss = loss[0:updation_per_epoch]
#fig = plt.figure()
p2, = plt.plot(steps, loss,'g',label='validation loss')
plt.grid()
# plt.xticks([w*updation_per_epoch for w in range(total_epoch)], ['%i'%w for w in range(total_epoch)])
# fig.suptitle('Validation Loss across {0} parameter updates'.format(int(updation_per_epoch*20)), fontsize=15)
# plt.xlabel('Epoch', fontsize=16)
# plt.ylabel('Validation Loss', fontsize=16)
# plt.axvspan(sr_epoch*updation_per_epoch, total_epoch*updation_per_epoch, facecolor='r', alpha=0.2,zorder=-1000)
# extra = Rectangle((0, 0), 1, 1, fc="#F08080", fill=True, edgecolor='#F08080', linewidth=1)
plt.legend([p1,p2], ["Training Loss","Validation Loss"],loc='upper right')
#plt.legend([p2], ["Validation Loss"],loc='upper right')
plt.show()