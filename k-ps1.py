import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

%matplotlib inline

# 1 question
data = sns.load_dataset("titanic")
data.loc[:10,:]

sns.boxplot(x="class",y="age",data=data)
sm.qqplot(data["age"])
plt.show()
sns.histplot(x="age",data=data)
plt.show()
plt.scatter(data["age"],data["fare"],s=20+100*data["survived"],c=data["pclass"],cmap="viridis",alpha=0.2)
plt.xlabel("Age(years)")
plt.ylabel("Fare")
plt.show()

# 2 question
phone_usage = [17, 23, 35, 36, 51, 53, 54, 55, 60, 77, 110]
n = len(phone_usage)
median = phone_usage[n//2]
Q2 = n//2 ; Q1 = Q2//2 ; Q3 = Q2+Q1+1
Q1, Q2, Q3 = phone_usage[Q1], phone_usage[Q2], phone_usage[Q3] 
IQR = (Q3 - Q1)
print("Median =",median,"\nIQR =",IQR)

# 3 question
outliers = []
for i in phone_usage:
    if (i > (Q3 + 1.5 * IQR)) or (i < (Q1 - 1.5 * IQR)):
        outliers.append(i)
print("The outliers are : ",outliers)

plt.boxplot(phone_usage,vert = False,patch_artist=True)
plt.xlabel("Length of calls(minutes)")
plt.show()
