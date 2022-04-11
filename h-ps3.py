# Question 1
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

# Question 2

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

# Question 3
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
