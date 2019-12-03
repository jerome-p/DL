import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd

style.use("ggplot")

log = pd.read_csv('model (3).log', header=None)
log.head()

epochs = log.iloc[:,2]
acc = log.iloc[:,3]
loss = log.iloc[:,4]
val_acc = log.iloc[:,5]
val_loss = log.iloc[:,6]

fig = plt.figure()

ax1 = plt.subplot2grid((2,1), (0,0))
ax2 = plt.subplot2grid((2,1), (1,0))

ax1.plot(log.iloc[:,1], acc, label='acc')
ax1.plot(log.iloc[:,1], val_acc, label='val_acc')
ax1.legend(loc=2)

ax2.plot(log.iloc[:,1], loss, label='loss')
ax2.plot(log.iloc[:,1], val_loss, label='val_loss')
ax2.legend(loc=2)

plt.show()


