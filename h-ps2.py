import pandas as pd

# Question 1

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


# Question 2
from collections import defaultdict
import pandas as pd

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

def minkowski_distance(X,Y):
    distance = 0
    for i in range(len(X)):
        distance += pow(abs(X[i]-Y[i]), 3)
    return pow(distance, 1/3)

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
    
# Question 3
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
