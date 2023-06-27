import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def scaleOut(file):
    yVector = []
    with open('sim0/DATAY/' + file) as yUnit:
        for line in yUnit:
            arr = line.split()
            yVector = [ float(arr[i]) for i in range(1, len(arr),2)]
            yVector[0]*= 10**2
            yVector[1]*= 10**5
    return yVector


def nonzeroGroups(df, cut = 2600):
#    ax = plt.axes(projection='3d')
#    x = df.loc['Position']['a'].values
#    y = df.loc['Position']['b'].values
#    z = df.loc['Position']['c'].values
#    ax.plot3D(x,y,z)
#    plt.show()
    df.loc['Position'] = df.loc['Position'].div(30)    
             
    groups = df.groupby(np.arange(len(df)) // 4)
    nzgroups = groups.apply(lambda x: x if x.loc['CtrlData']['a'] != 0.0 else None).values
    np.reshape(nzgroups,[-1])
    nzgroups = nzgroups[~np.isnan(nzgroups)]
    return nzgroups[:cut*13]

resX= os.listdir("sim0/DATAX/")
resY= os.listdir("sim0/DATAY/")
xtrainData = []
ytrainData = []
xtestData = []
ytestData = []

TEST_SIZE = 30
TRAIN_SIZE = 230

iter = 0
for file in resX:
    print(file)
    df = pd.read_csv('sim0/DATAX/'+ file,sep=' ', header=None, names=['a','b','c','d'])
    xVector = nonzeroGroups(df)
    yVector = scaleOut(file)
    if iter < TEST_SIZE:
        xtestData.append(xVector)
        ytestData.append(yVector)
    elif iter < TRAIN_SIZE:
        xtrainData.append(xVector)
        ytrainData.append(yVector)
    else:
        iter = 0
        break
    iter += 1



train_dataset = tf.data.Dataset.from_tensors((xtrainData,ytrainData))
test_dataset = tf.data.Dataset.from_tensors((xtestData,ytestData))


print(train_dataset)
print(test_dataset)

train_dataset.save('sim0/Train/')
#val_dataset.save('sim0/Val/')
test_dataset.save('sim0/Test/')
print(train_dataset)