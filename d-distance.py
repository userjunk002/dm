#EUCLEDIAN DISTANCE MATRIX
import pandas as pd
d2={"team":[5,3,0,0],"coach":[0,0,7,1],"hockey":[3,2,0,0,],"baseball":[0,0,2,0],"soccer":[2,1,1,1],
   "penalty":[0,1,0,2],"score":[0,0,0,2],"win":[2,1,3,0],"loss":[0,0,0,3],"season":[0,1,0,0]}
data2=pd.DataFrame(d2)
print(data2)
xy_list = np.array(data2)
dist = lambda p1, p2: math.sqrt(sum([(p1[i]-p2[i])**2 for i in range(len(p1))]))
dm = np.asarray([[dist(p1, p2) for p2 in xy_list] for p1 in xy_list])

#COSINE SIMILARITY MATRIX
import numpy as np
from numpy.linalg import norm
dist = lambda p1, p2: np.dot(p1, p2)/(norm(p1)*norm(p2))
dm = np.asarray([[dist(p1, p2) for p2 in xy_list] for p1 in xy_list])
print(dm)

#MANHATTAN DISTANCE MATRIX
def manhattan_distance(a, b):
    return np.abs(a - b).sum()
dm = np.asarray([[manhattan_distance(p1, p2) for p2 in xy_list] for p1 in xy_list])
print(dm)

#MINKOWSKI DISTANCE MATRIX
from math import *
from decimal import Decimal
 
def p_root(value, root):
     
    root_value = 1 / float(root)
    return float(round (Decimal(value) **
             Decimal(root_value), 3))
 
def minkowski_distance(x, y, p_value):
    return (p_root(sum(pow(abs(a-b), p_value)
            for a, b in zip(x, y)), p_value))

p = 3
dm = np.asarray([[minkowski_distance(p1, p2,p) for p2 in xy_list] for p1 in xy_list])

#CHEBYSHEV SUPREMUM DISTANCE MATRIX
import numpy as np
import math

def chebyshev(u, v):
    return max(abs(u - v))
dm = np.asarray([[chebyshev(p1, p2) for p2 in xy_list] for p1 in xy_list])
print(dm)

#JACCARD DISTANCE MATRIX

import numpy as np
def jaccard(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    intersection = np.logical_and(im1, im2)
    union = np.logical_or(im1, im2)
    return intersection.sum() / float(union.sum())
xy_list2=data3.to_numpy()
dm = np.asarray([[jaccard(p1, p2) for p2 in xy_list2] for p1 in xy_list2])
print(1-dm)

#COVARIANCE
x=[1, 1, 1, 1]
y=[2, 2, 2, 2]
mx=np.mean(x)
my=np.mean(y)
print(mx,my)
ax=[i-mx for i in x]
ay=[i-my for i in y]
print(ax,ay)
sum([ax[i]*ay[i] for i in range(len(ax))])/len(ax)

#EUCLEDIAN
math.sqrt(sum([(y[i]-x[i])^2 for i in range(len(x))]))

#HAMMING

def hammingDistance(x, y):
    if type(x) and type(y)==str:
        if len(x)==len(y):
            hamming=0
            i = 0
            s1, s2 = [], []
            [[s1.append(ord(val))] for val in x] # the ord function returns the ASCII
            # equivalent of a letter
            [[s2.append(ord(val))] for val in y]
            for x in s1:
                if x!=s2[i]: # assumption, strings are of equal length
                    hamming+=1
                i+=1
            return hamming
        else:
            return "Strings not of equal length"

    elif type(x) and type(y)==int:
        return bin(x ^ y).count('1')
    else:
        return "Unknown format of inputs"
    # bin converts int to binary string
    # x^y is a logical operator which produces 0 when digits in corresponding positions match
    # and 1 if otherwise

print(hammingDistance(2,10))
protein_sequence1="GAATGCAGATGGACTCTAGA"
protein_sequence2="GAATAGCTAATCACTCTAGA"
print(hammingDistance(protein_sequence1,protein_sequence2))
print("There is a",
      int((len(protein_sequence1)-hammingDistance(protein_sequence1,protein_sequence2))/len(protein_sequence1)*100),
"% match between the 2 protein sequences")


#BINNING

import numpy as np

b=np.array([13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70])
#b=np.sort(b)  #sort the array
  
# create bins
bin1=np.zeros((9,3)) 
bin2=np.zeros((9,3))
bin3=np.zeros((9,3))
  
# Bin mean
for i in range (0,27,3):
    k=int(i/3)
    mean=(b[i] + b[i+1] + b[i+2])/3
    for j in range(3):
        bin1[k,j]=mean
print("Bin Mean: \n",bin1)
     
# Bin boundaries
for i in range (0,27,3):
    k=int(i/3)
    for j in range (3):
        if (b[i+j]-b[i]) < (b[i+2]-b[i+j]):
            bin2[k,j]=b[i]
        else:
            bin2[k,j]=b[i+2]       
print("Bin Boundaries: \n",bin2)
  
# Bin median
for i in range (0,27,3):
    k=int(i/3)
    for j in range (3):
        bin3[k,j]=b[i+1]
print("Bin Median: \n",bin3)

print("End")

#Z-SCORE
values = [4,5,6,6,6,7,8,12,13,13,14,18]

mean = sum(values) / len(values)
differences = [(value - mean)**2 for value in values]
sum_of_differences = sum(differences)
standard_deviation = (sum_of_differences / (len(values) - 1)) ** 0.5

print(standard_deviation)

zscores = [(value - mean) / standard_deviation for value in values]

print(zscores)

#CORRELATION COEFFICIENT

x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
a=[2, 3, 3, 4, 6, 7, 8, 9, 10, 11]
b=[15, 13, 12, 12, 10, 10, 8, 7, 4, 3]
def mean(x):
    return round(sum(x)/len(x),1)

def standardDeviation(values,mean):
    data=[(val-mean)**2 for val in values]
    return (sum(data)/float(len(data)))**0.5


def pearsonCorrelationCoefficient(x,y):
    xMean=mean(x)
    yMean=mean(y)
    xStd=standardDeviation(x,xMean)
    yStd=standardDeviation(y,yMean)
    numerator = sum( (x[i]-xMean)*(y[i]-yMean) for i in range(len(x)))
    denominator = len(x)*xStd*yStd
    return round((numerator/denominator),3)

print(pearsonCorrelationCoefficient(x,y))
print(pearsonCorrelationCoefficient(x,a))
print(pearsonCorrelationCoefficient(x,b))


# Positive Correlation: both variables change in the same direction.
# Neutral Correlation: No relationship in the change of the variables.
# Negative Correlation: variables change in opposite directions.

#MANHATTAN

from math import sqrt

#create function to calculate Manhattan distance 
def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))
 
#define vectors
A = [2, 4, 4, 6]
B = [5, 5, 7, 8]

#calculate Manhattan distance between vectors
manhattan(A, B)


#JACCARD

import pandas as pd
d3={"fever":['Y','Y','Y'],"cough":['N','Y','N'],"test-1":['P','N','P'],
    "test2":['N','N','N'],"test3":['N','N','P'],"test4":['N','N','N']}
data3=pd.DataFrame(d3)
print(data3)

#%%
data3["fever"][data3["fever"]=='Y'] = 1
data3["cough"][data3["cough"]=='Y'] = 1
data3["cough"][data3["cough"]=='N'] = 0
data3["test-1"][data3["test-1"]=='P'] = 1
data3["test-1"][data3["test-1"]=='N'] = 0
data3["test2"][data3["test2"]=='N'] = 0
data3["test3"][data3["test3"]=='N'] = 0
data3["test3"][data3["test3"]=='P'] = 1
data3["test4"][data3["test4"]=='N'] = 0
print(data3)

#%%
from sklearn.metrics import pairwise_distances
ans3=pairwise_distances(data3.to_numpy(), metric="jaccard")
print(ans3)


#DISSIMILARITY MATRIX
import pandas as pd

d2={"team":[5,3,0,0],"coach":[0,0,7,1],"hockey":[3,2,0,0,],"baseball":[0,0,2,0],"soccer":[2,1,1,1],
   "penalty":[0,1,0,2],"score":[0,0,0,2],"win":[2,1,3,0],"loss":[0,0,0,3],"season":[0,1,0,0]}
data2=pd.DataFrame(d2)
print(data2)

#%%
from sklearn.metrics import pairwise_distances
ans1=pairwise_distances(data2,metric="cosine")
print(ans1)
#%%
ans2=pairwise_distances(data2, metric="euclidean")
print(ans2)
#%%
ans3=pairwise_distances(data2, metric="manhattan")
print(ans3)
#%%
ans4=pairwise_distances(data2, metric="chebyshev")
print(ans4)
#%%
ans5=pairwise_distances(data2, metric="minkowski",p=3)
print(ans5)
