# Q1
euclid_dist = lambda x,y: sum((x[i]-y[i])**2 for i in range(len(x)))**0.5
euclid_dist.__name__ = "Euclidean distance"
manhat_dist = lambda x,y: sum(abs(x[i]-y[i]) for i in range(len(x)))
manhat_dist.__name__ = "Manhattan distance"
supremum_dist = lambda x,y: max([abs(x[i]-y[i]) for i in range(len(x))]+[0])
supremum_dist.__name__ = "Supremum distance"
cosine_simil = lambda x,y: sum(x[i]*y[i] for i in range(len(x)))/(sum(i*i for i in x)*sum(i*i for i in y))**0.5
cosine_simil.__name__ = "Cosine similarity"
minkow_dist3 = lambda x,y: pow(sum(abs(x[i]-y[i])**3 for i in range(len(x))),1/3)
minkow_dist3.__name__ = "Minkowski distance(h=3)"

data = [[1.5, 1.7],
        [2, 1.9],
        [1.6, 1.8],
        [1.2, 1.5],
        [1.5,1.0]]

query = [1.4,1.6]
print("\nQ1 :\n")
print("Similarity rank of query based on\n")
print("Euclidean dist :",sorted(data, key = lambda x: euclid_dist(x,query)))
print("\nManhattan dist :",sorted(data, key = lambda x: manhat_dist(x,query)))
print("\nSupremum dist :",sorted(data, key = lambda x: supremum_dist(x,query)))
print("\nCosine Similarity :",sorted(data, key = lambda x: cosine_simil(x,query)))

#Q2
docs = [[5, 0, 3, 0, 2, 0, 0, 2, 0, 0],
        [3, 0, 2, 0, 1, 1, 0, 1, 0, 1],
        [0, 7, 0, 2, 1, 0, 0, 3, 0, 0],
        [0, 1, 0, 0, 1, 2, 2, 0, 3, 0]]
def matrix(measure,docs):
    nDocs = len(docs)
    simil_matrix = [[1]*nDocs for i in range(nDocs)]
    for i in range(nDocs):
        for j in range(i,nDocs):
            simil_matrix[i][j] = simil_matrix[j][i] = "{0:.2f}".format(measure(docs[i],docs[j]))
    print("\n"+measure.__name__,"Matrix\n   D1   D2   D3   D4")
    for i in range(nDocs):
        print("D"+str(i+1),*simil_matrix[i])
    return simil_matrix

print("\nQ2 :")
matrix(cosine_simil,docs)
matrix(euclid_dist,docs)
matrix(manhat_dist,docs)
matrix(minkow_dist3,docs)
matrix(supremum_dist,docs)
print("\n")


#Q3
def binary_dissimilarity(x,y):
    q=r=s=t=0
    for i in range(len(x)):
        if x[i]==y[i]:
            if x[i]==1:
                q+=1
            else:
                t+=1
        else:
            if x[i]==1:
                r+=1
            else:
                s+=1
    return (r+s)/(q+r+s)
data = [[1,1,0,1,0,0,0],
        [1,1,1,0,0,0,0],
        [0,1,0,1,0,1,0]]
name = ["Jack","Jim","Mary"]
n = len(data)
print("\nQ3 :\n")
for i in range(n):
    for j in range(i+1,n):
        dist = binary_dissimilarity(data[i],data[j])
        print("Dissimilarity of "+name[i]+" and "+name[j]+" is",dist)
