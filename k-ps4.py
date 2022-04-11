euclid_dist = lambda x,y: sum((x[i]-y[i])**2 for i in range(len(x)))**0.5
manhat_dist = lambda x,y: sum(abs(x[i]-y[i]) for i in range(len(x)))
supremum_dist = lambda x,y: max([abs(x[i]-y[i]) for i in range(len(x))]+[0])
cosine_simil = lambda x,y: sum(x[i]*y[i] for i in range(len(x))) / (sum(i*i for i in x)*sum(i*i for i in y))
hamming = lambda x,y:sum(x[i]!=y[i] for i in range(len(x)))

def jaccard(x,y):
    q=r=s=t=0
    for i in range(len(x)):
        if x[i]==y[i]:
            if x[i] in [1,"1"]:
                q+=1
            else:
                t+=1
        else:
            if x[i] in [1,"1"]:
                r+=1
            else:
                s+=1
    return (q)/(q+r+s)

def correlation(X,Y):
    N = len(X)
    x_mean, y_mean = sum(X)/N, float(str(sum(Y)/N)[:9])
    s_xy, s_xx, s_yy = 0, 0, 0
    for i in range(N):
        s_xy += (X[i]-x_mean)*(Y[i]-y_mean)
        s_xx += (X[i]-x_mean)**2
        s_yy += (Y[i]-y_mean)**2
    try :
        r = s_xy / (s_xx*s_yy)**0.5
    except:
        r = 0
    return r

# Q1
x = [1,1,1,1] ; y = [2,2,2,2]
print("Cosine simil :",cosine_simil(x,y),"| Correlation :",correlation(x,y),"| Euclidean :",euclid_dist(x,y))
x = [0,1,0,1] ; y = [1,0,1,0]
print("\nCosine simil :",cosine_simil(x,y),"| Correlation :",correlation(x,y),"| Euclidean :",euclid_dist(x,y),"| Jaccard : ",jaccard(x,y))
x = [0,-1,0,1] ; y = [1,0,-1,0]
print("\nCosine simil :",cosine_simil(x,y),"| Correlation :",correlation(x,y),"| Euclidean :",euclid_dist(x,y))
x = [1,1,0,1,0,1] ; y = [1,1,1,0,0,1]
print("\nCosine simil :",cosine_simil(x,y),"| Correlation :",correlation(x,y),"| Jaccard :",jaccard(x,y))
x = [2,-1,0,2,0,-3] ; y = [-1,1,-1,0,0,-1]
print("\nCosine simil :",cosine_simil(x,y),"| Correlation :",correlation(x,y))

# Q2
X="0101010001"
Y="0100011000"
print("Jaccard :",jaccard(X,Y),"| Hamming Distance :",hamming(X,Y))