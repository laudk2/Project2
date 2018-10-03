import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import random
ni = 1000
epsilon = 0.00005

maleAverageHeight = 5.78
maleAverageWeight = 195.8
maleHeight = []
maleWeight = []
convertMaleHeight = [];
convertMaleWeight = [];

femaleAverageHeight = 5.34
femaleAverageWeight = 168.4
femaleHeight = []
femaleWeight = []
convertFemaleHeight = [];
convertFemaleWeight = [];

separationLineA = pd.read_csv("sep_line_a.txt", header=None)
separationLineB = pd.read_csv("sep_line_b.txt", header=None)


file = open('SampleData.txt', 'w')
for x in range(0, 2):
    for y in range(0, 2000):
        if x == 0:
            height = np.random.normal(maleAverageHeight, 0.3)
            weight = np.random.normal(maleAverageWeight, 20)
            maleHeight.append(height)
            maleWeight.append(weight)
            file.write(str(maleHeight[y]) + "," + str(maleWeight[y]) + "," + str(x) + "\n")
        else:
            height = np.random.normal(femaleAverageHeight, 0.3)
            weight = np.random.normal(femaleAverageWeight, 20)
            femaleHeight.append(height)
            femaleWeight.append(weight)
            file.write(str(femaleHeight[y]) + "," + str(femaleWeight[y]) + "," + str(x) + "\n")
file.close()

def nomalize(Data):


    copyData = Data.copy()

    minHeight = Data[0].min()
    maxHeight = Data[0].max()
    minWeight = Data[1].min()
    maxWeight = Data[1].max()

    global convertMaleHeight
    global convertMaleWeight
    global convertFemaleHeight
    global convertFemaleWeight

    for x in range (0,2000):
        convertMaleHeight.append((maleHeight[x] - minHeight)/(maxHeight - minHeight))
    for y in range (0,2000):
        convertMaleWeight.append((maleWeight[y] - minWeight)/(maxWeight - minWeight))
    for g in range (0,2000):
        convertFemaleHeight.append((femaleHeight[g] - minHeight)/(maxHeight - minHeight))
    for z in range (0,2000):
        convertFemaleWeight.append((femaleWeight[z] - minWeight)/(maxWeight - minWeight))



    print(convertFemaleHeight)

    copyData[0] = (Data[0] - minHeight)/ (maxHeight - minHeight)
    copyData[1] = (Data[1] - minWeight)/ (maxWeight - minWeight)

    return copyData
def plot_xy_sep_line(sep_line, data_frame, color="0.18"):
    x_weight = sep_line[0]
    y_weight = sep_line[1]
    bias = sep_line[2]

    min = data_frame[0].min()
    max = data_frame[0].max()
    mid = (min + ((max - min) / 2))

    # formula is y_weight(y) = x_weight(x) + bias(1)
    # or y = (x_weight/a)y_weight + (bias/y_weight)
    y1 = -(((x_weight * min) / y_weight) + (bias / y_weight))
    y2 = -(((x_weight * max) / y_weight) + (bias / y_weight))
    y_mid = -(((x_weight * mid) / y_weight) + (bias / y_weight))

    ax = plt.axes()
    ax.arrow(mid, y_mid, 0.05, 0.05, head_width=0.025, head_length=0.025, fc='k', ec='k', color="b")

    plt.plot([min, max], [y1, y2], color=color)

    return plt

hardActGraph = pd.read_csv("SampleData.txt", header=None)


hardActGraph = nomalize(hardActGraph)
print(hardActGraph)
trainGraph = hardActGraph.sample(frac=0.25)
testGraph = hardActGraph[~hardActGraph.isin(trainGraph)]



plt.figure(1)

maleGraphB = plt.scatter(convertMaleHeight, convertMaleWeight, alpha=.25, s=35, label='Male')
femaleGraphB = plt.scatter(convertFemaleHeight, convertFemaleWeight, alpha=.25, s=35, label='Female')
plt.legend(loc='upper right')
plt.xlabel("Height(ft)")
plt.ylabel("Weight(lbs)")

randX = random.uniform(0.1,0.9)
SepLine = [-randX, - randX, randX]
plot_xy_sep_line(SepLine,hardActGraph,color="g")


plt.show()

print("hello world")