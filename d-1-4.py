# Boxplots
# Quantile plot
# Quantile- Quantile plot
# Histograms
# Scatter Plots 

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

data = sns.load_dataset("iris")

sns.boxplot(x = data["species"], y = data["sepal_length"])
plt.show()

sns.distplot(x = data["petal_length"], hist=True, kde=False, rug=False)
plt.show()

sns.scatterplot(x = data["sepal_length"], y = data["sepal_width"])
plt.show()

rd = pd.DataFrame(np.array(data["sepal_length"].tolist()))
res = rd.describe().T.drop('count', axis=1)
plt.plot([res["min"], res["25%"], res["50%"], res["75%"], res["max"]], [0,0.25,0.5,0.75,1])
plt.show()

sm.qqplot( data["sepal_length"])
plt.show()

#INTER QUANTILE RANGE 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = [17, 23, 35, 36, 51, 53, 54, 55, 60, 77, 110]
quartiles = np.percentile(data, [25,50,75])
print(f"Median = {np.median(data)}\nQuartiles = {quartiles}")

inter_quantile_region = quartiles[2] - quartiles[0]
low = quartiles[0] - 1.5*inter_quantile_region
high = quartiles[2] + 1.5*inter_quantile_region
outliers = []
for i in data:
  if i > high or i < low:
    outliers.append(i)
print(f"Outliers in dataset: {outliers}")

sns.boxplot(data)
plt.show()
sns.boxplot([6,12,15,17,22])
plt.show()

#DATA OUTLIER

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = [6, 25, 39, 62, 65, 74, 80, 94, 125, 127, 154, 159, 184, 210, 251]
quartiles = np.percentile(data, [25,50,75])
print(f"Median = {np.median(data)}\nQuartiles = {quartiles}")

inter_quantile_region = quartiles[2] - quartiles[0]
low = quartiles[0] - 1.5*inter_quantile_region
high = quartiles[2] + 1.5*inter_quantile_region
outliers = []
for i in data:
  if i > high or i < low:
    outliers.append(i)
print(f"Outliers in dataset: {outliers}")

sns.boxplot(data)
plt.show()

#DISTANCES

# x1,1.5,1.7
# x2,2,1.9
# x3,1.6,1.8
# x4,1.2,1.5
# x5,1.5,1.0

import pandas as pd
from collections import defaultdict

def manhattan_distance(X,Y):
    distance = 0
    for i in range(len(X)):
        distance += abs(X[i]-Y[i])
    return distance

def euclidean_distance(X,Y):
    distance = 0
    for i in range(len(X)):
        distance += pow(abs(X[i]-Y[i]), 2)
    return pow(distance, 1/2)

def supremum_distance(X,Y):
    distance = 0
    for i in range(len(X)):
        distance = max(abs(X[i]-Y[i]), distance)
    return distance

def cosine_similarity(X,Y):
    sum_xi_yi = 0
    sum_xi_2 = 0
    sum_yi_2 = 0
    for i in range(len(X)):
        sum_xi_yi += X[i]*Y[i]
        sum_xi_2 += pow(X[i], 2)
        sum_yi_2 += pow(Y[i], 2)
    return sum_xi_yi/(pow(sum_xi_2, 1/2) * pow(sum_yi_2, 1/2))

df = pd.read_csv("./data.csv", names=['Name', 'A1', 'A2'])
data_point = {'Name':'x6', 'A1':1.4, 'A2':1.6}

manhattan_results, euclidean_results, supremum_results, cosine_results = {}, {}, {}, {}

for index, point in df.iterrows():
    manhattan_results[point['Name']] = manhattan_distance( [point['A1'], point['A2']], [data_point['A1'], data_point['A2']])
    euclidean_results[point['Name']] = euclidean_distance( [point['A1'], point['A2']], [data_point['A1'], data_point['A2']])
    supremum_results[point['Name']] = supremum_distance( [point['A1'], point['A2']], [data_point['A1'], data_point['A2']])
    cosine_results[point['Name']] = cosine_similarity( [point['A1'], point['A2']], [data_point['A1'], data_point['A2']])

max_manhattan_result = max(manhattan_results, key= lambda x: manhattan_results[x])
print(f"\nManhattan Results:\n{manhattan_results}\nMax val: {max_manhattan_result}")

max_euclidean_result = max(euclidean_results, key= lambda x: euclidean_results[x])
print(f"\nEuclidean Results:\n{euclidean_results}\nMax val: {max_euclidean_result}")

max_supremum_result = max(supremum_results, key= lambda x: supremum_results[x])
print(f"\nSupremum Results:\n{supremum_results}\nMax val: {max_supremum_result}")

max_cosine_result = max(cosine_results, key= lambda x: cosine_results[x])
print(f"\nCosine Results:\n{cosine_results}\nMax val: {max_cosine_result}\n\n")



#DOCUMENT DISTANCES


# Document1,5,0,3,0,2,0,0,2,0,0
# Document2,3,0,2,0,1,1,0,1,0,1
# Document3,0,7,0,2,1,0,0,3,0,0
# Document4,0,1,0,0,1,2,2,0,3,0


df = pd.read_csv("./data.csv", names=['Document', 'Team', 'Coach', 'Hockey', 'Baseball', 'Soccer', 'Penalty', 'Score', 'Win', 'Loss', 'Season'])
manhattan_results, euclidean_results, supremum_results, cosine_results, minkowski_results = defaultdict(dict),defaultdict(dict),defaultdict(dict),defaultdict(dict), defaultdict(dict)

for i,x in df.iterrows():
    for j,y in df.iterrows():
        manhattan_results[x['Document']][y['Document']] = manhattan_distance([ x[k] for k in x.keys() if k != 'Document'], [ y[k] for k in y.keys() if k != 'Document'])
        euclidean_results[x['Document']][y['Document']] = euclidean_distance([ x[k] for k in x.keys() if k != 'Document'], [ y[k] for k in y.keys() if k != 'Document'])
        supremum_results[x['Document']][y['Document']] = supremum_distance([ x[k] for k in x.keys() if k != 'Document'], [ y[k] for k in y.keys() if k != 'Document'])
        cosine_results[x['Document']][y['Document']] = cosine_similarity([ x[k] for k in x.keys() if k != 'Document'], [ y[k] for k in y.keys() if k != 'Document'])
        minkowski_results[x['Document']][y['Document']] = minkowski_distance([ x[k] for k in x.keys() if k != 'Document'], [ y[k] for k in y.keys() if k != 'Document'])


#print(f"{manhattan_results}\n\n{euclidean_results}\n\n{supremum_results}\n\n{cosine_results}\n\n{minkowski_results}")

print(f"\nManhattan results")
for key in manhattan_results.keys():
    print(f"{key} -> {manhattan_results[key]}")

print(f"\nEuclidean results")
for key in euclidean_results.keys():
    print(f"{key} -> {euclidean_results[key]}")

print(f"\nSupremum results")
for key in supremum_results.keys():
    print(f"{key} -> {supremum_results[key]}")

print(f"\nCosine results")
for key in cosine_results.keys():
    print(f"{key} -> {cosine_results[key]}")

print(f"\nMinkowski results")
for key in minkowski_results.keys():
    print(f"{key} -> {minkowski_results[key]}")

#Dissimilarity between binary attributes:

# Jack,M,Y,N,P,N,N,N
# Jim,M,Y,Y,N,N,N,N
# Mary,F,Y,N,P,N,P,N

from collections import defaultdict
import pandas as pd

def getState(x):
    if x == "P" or x == "Y":
        return "True"
    return "False"

def get_binary_attribute_dissimilarity(X,Y):
    matched = defaultdict(dict)
    matched["True"]["True"] = 0
    matched["True"]["False"] = 0
    matched["False"]["True"] = 0
    matched["False"]["False"] = 0
    for xi,yi in zip(X,Y):
        matched[getState(xi)][getState(yi)] += 1
    # print(matched)
    return (matched["True"]["False"]+matched["False"]["True"]) / (matched["True"]["True"]+matched["True"]["False"]+matched["False"]["True"])


df = pd.read_csv("./data.csv", names=['Name', 'Gender', 'Fever', 'Cough', 'Test-1', 'Test-2', 'Test-3', 'Test-4'])
similarity_results = defaultdict(dict)
del df['Gender']

for i,x in df.iterrows():
    for j,y in df.iterrows():
        if x['Name'] != y['Name']:
            similarity_results[x['Name']][y['Name']] = get_binary_attribute_dissimilarity([ x[k] for k in x.keys() if k != 'Name'], [ y[k] for k in y.keys() if k != 'Name'])

print(f"Dissimilarity results")
for key in similarity_results.keys():
    print(f"{key} -> {similarity_results[key]}")


#BINNING

import numpy as np

def bin_smoothing_mean(data, bin):
    i = 0
    result = []
    while i < len(data):
        j = 0
        bin_data = []
        while j < bin:
            bin_data.append(data[i+j])
            j += 1
        i += j+1
        result.extend([np.mean(bin_data) for num in bin_data])
    return result

def bin_smoothing_median(data, bin):
    i = 0
    result = []
    while i < len(data):
        j = 0
        bin_data = []
        while j < bin:
            bin_data.append(data[i+j])
            j += 1
        i += j + 1
        result.extend([np.median(bin_data) for i in bin_data])
    return result

def bin_smoothing_boundaries(data, bin):
    i = 0
    result = []
    while i < len(data):
        j = 0
        bin_data = []
        while j < bin:
            bin_data.append(data[i+j])
            j += 1
        i += j + 1
        
        for num in bin_data:
            if abs(num-min(bin_data)) < abs(num-max(bin_data)):
                result.append(min(bin_data))
            else:
                result.append(max(bin_data))

    return result



data = [13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70]
print(f"Mean smoothing : {bin_smoothing_mean(data,3)}\n\nMedian smoothing : {bin_smoothing_median(data,3)}\n\nBorder Smoothing : {bin_smoothing_boundaries(data,3)}")


#Z SCORE NORMALIZE

# 23,9.5
# 23,26.5
# 27,7.8
# 27,17.8
# 39,31.4
# 41,25.9
# 47,27.4
# 49,27.2

from cmath import sqrt
import numpy as np
import pandas as pd

def z_score_normalization(data: pd.DataFrame)-> pd.DataFrame:
    normalized_data = {}
    mean_values = {}
    standard_deviation_values = {}

    for column in data.columns:
        normalized_data[column] = []
        mean_values[column] = np.mean(data[column].tolist())
        standard_deviation_values[column] = np.std(data[column].tolist())

    for index, row in data.iterrows():
        for column in data.columns:
            result = (row[column] - mean_values[column]) / standard_deviation_values[column]
            normalized_data[column].append(result)
    return pd.DataFrame(normalized_data)

def calc_pearson_correlation_coefficient(X: list,Y: list)-> float:
    N = len(X)
    x_mean, y_mean = np.mean(X), np.mean(Y)
    s_xy, s_xx, s_yy = 0, 0, 0
    for i in range(N):
        s_xy += (X[i]-x_mean)*(Y[i]-y_mean)
        s_xx += (X[i]-x_mean)**2
        s_yy += (Y[i]-y_mean)**2
    result = s_xy / (s_xx*s_yy)**0.5
    return result


data = pd.read_csv("data.csv", names = ["Age", "Fat"])
normalized_data = z_score_normalization(data)
print(f"Normalized data based on Z-score : \n{normalized_data}")
res = calc_pearson_correlation_coefficient( normalized_data["Age"].tolist(), normalized_data["Fat"].tolist() )
print(f"\nPearson Coefficient between Age and Fat is : { res } ")



#GENERATING HIERARCHY

from tkinter import N

import numpy


def get_hierarchy_for_nominal_data(columns: list)-> dict:
    # columns = [{ name: 'Column Name', data: [] (values in that column in array format) }]
    concept_hierarchy_map = {} # stores hierarchies of all nominal columns
    for columnar_data in columns:
        value_freq_map = {} # stores values and their frequencies belonging to a column
        for data in columnar_data.data:
            if data in value_freq_map:
                value_freq_map[data] += 1
            else:
                value_freq_map[data] = 1
        # stores column name in ascending order of their frequencies
        sorted_results = sorted(value_freq_map.keys() , key=lambda k: value_freq_map[k])
        # Column name and the hierarchy for it are mapped and stored
        concept_hierarchy_map[columnar_data.name] = sorted_results
    return concept_hierarchy_map

def get_hierarchy_for_numeric_data(data: list, no_of_intervals: int)-> dict:
    concept_hierarchy = {}
    concept_hierarchy["label"] = str(min(data)) + "-" +  str(max(data))
    concept_hierarchy["value"] = []
    
    bin_size = ( max(data) - min(data) + 1 ) // no_of_intervals
    print(f"Width of bin: {bin_size}")

    iter = min(data)
    while iter < max(data):
        result = []
        for value in data:
            if iter <= value < iter+bin_size:
                result.append(value)
        concept_hierarchy["value"].append({
            "label": str(iter)+"-"+str(iter+bin_size-1),
            "count": len(result),
            "mean": numpy.mean(result),
            "sum": sum(result)
        })
        iter += bin_size

    return concept_hierarchy

print( get_hierarchy_for_numeric_data([ 1, 1, 5, 5, 5,
5, 5, 8, 8, 10, 10, 10, 10, 12, 14, 14, 14, 15, 15, 15, 15, 15, 15, 18, 18, 18, 18, 18,
18, 18, 18, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 25, 25, 25, 25, 25, 28, 28, 30,
30, 30], 3) )

# SIMILARITIES AND DISTANCES

import math
from typing import List
import numpy as np

def cosine_similarity(X: List,Y: List)-> float:
    sum_xi_yi = 0
    sum_xi_2 = 0
    sum_yi_2 = 0
    for i in range(len(X)):
        sum_xi_yi += X[i]*Y[i]
        sum_xi_2 += pow(X[i], 2)
        sum_yi_2 += pow(Y[i], 2)
    return sum_xi_yi/(pow(sum_xi_2, 1/2) * pow(sum_yi_2, 1/2))

def euclidean_distance(X: List,Y: List)-> float:
    distance = 0
    for i in range(len(X)):
        distance += pow(abs(X[i]-Y[i]), 2)
    return pow(distance, 1/2)

def pearson_correlation_coefficient(X: List,Y: List)-> float:
    N = len(X)
    x_mean, y_mean = np.mean(X), np.mean(Y)
    s_xy, s_xx, s_yy = 0, 0, 0
    for i in range(N):
        s_xy += (X[i]-x_mean)*(Y[i]-y_mean)
        s_xx += (X[i]-x_mean)**2
        s_yy += (Y[i]-y_mean)**2
    result = s_xy / (s_xx*s_yy)**0.5
    if math.isnan(result):
        result = 0.0
    return result

def jacardian_distance(X: List,Y: List)-> float:
    return len(set(X).intersection(set(Y)))/len(set(X).union(set(Y)))

q1_1, q1_2 = [1, 1, 1, 1], [2, 2, 2, 2] # cosine, correlation, Euclidean
q2_1, q2_2 = [0, 1, 0, 1], [1, 0, 1, 0] # cosine, correlation, Euclidean, Jaccard
q3_1, q3_2 = [0, -1, 0, 1], [1, 0, -1, 0] # cosine, correlation, Euclidean
q4_1, q4_2 = [1, 1, 0, 1, 0, 1] , [1, 1, 1, 0, 0, 1] # cosine, correlation, Jaccard
q5_1, q5_2 = [2, -1, 0, 2, 0, -3] , [-1, 1, -1, 0, 0, -1] #cosine, correlation

print(cosine_similarity(q1_1, q1_2), pearson_correlation_coefficient(q1_1,q1_2), euclidean_distance(q1_1,q1_2))
print(cosine_similarity(q2_1, q2_2), pearson_correlation_coefficient(q2_1,q2_2), euclidean_distance(q2_1,q2_2), jacardian_distance(q2_1, q2_2))
print(cosine_similarity(q3_1, q3_2), pearson_correlation_coefficient(q3_1,q3_2), euclidean_distance(q3_1,q3_2))
print(cosine_similarity(q4_1, q4_2), pearson_correlation_coefficient(q4_1,q4_2), jacardian_distance(q4_1,q4_2))
print(cosine_similarity(q5_1, q5_2), pearson_correlation_coefficient(q5_1,q5_2))

#HAMMING AND JACCARDIAN

def hamming_distance(X,Y):
    count = 0
    for i in range(len(X)):
        if X[i] != Y[i]:
            count += 1
    return count

def jacardian_distance(X,Y):
    return len(set(X).intersection(set(Y)))/len(set(X).union(set(Y)))

print(f"Hamming Distance\t: {hamming_distance('0101010001', '0100011000')}")
print(f"Jacardian Similarity\t: {jacardian_distance('0101010001', '0100011000')} ")