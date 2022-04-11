# Question 1
import numpy as np

def cosine_similarity(X,Y):
    sum_xi_yi = 0
    sum_xi_2 = 0
    sum_yi_2 = 0
    for i in range(len(X)):
        sum_xi_yi += X[i]*Y[i]
        sum_xi_2 += pow(X[i], 2)
        sum_yi_2 += pow(Y[i], 2)
    return sum_xi_yi/(pow(sum_xi_2, 1/2) * pow(sum_yi_2, 1/2))

def euclidean_distance(X,Y):
    distance = 0
    for i in range(len(X)):
        distance += pow(abs(X[i]-Y[i]), 2)
    return pow(distance, 1/2)

def pearson_correlation_coefficient(X,Y):
    N = len(X)
    x_mean, y_mean = np.mean(X), np.mean(Y)
    s_xy, s_xx, s_yy = 0, 0, 0
    for i in range(N):
        s_xy += (X[i]-x_mean)*(Y[i]-y_mean)
        s_xx += (X[i]-x_mean)**2
        s_yy += (Y[i]-y_mean)**2
    result = s_xy / (s_xx*s_yy)**0.5
    return result

def jacardian_distance(X,Y):
    return len(set(X).intersection(set(Y)))/len(set(X).union(set(Y)))

q1_1, q1_2 = [1, 1, 1, 1)], [2, 2, 2, 2]
q2_1, q2_2 = [0, 1, 0, 1], [1, 0, 1, 0]

# Question 2
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
