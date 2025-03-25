import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'movie': ['a','b','c','a','b','c','a','b','c'],
    'rating': [9,7,6,9,5,7,8,9,5]
})

# print(df.mean(numeric_only=True))
df.rating.mean()
sns.displot(df,kde=True,rug=True)
plt.axvline(np.mean(df.rating),color ='r',linestyle ='-')
plt.axvline(np.median(df.rating),color ='g',linestyle ='-')
plt.axvline(df.rating.mode().values[0], color='y', linestyle='-')
# plt.show()
f,(ax_box,ax_hist)=plt.subplots(2,sharex=True,gridspec_kw={'height_ratios':(0.2,1)})

mean =np.mean(df.rating)
median = np.median(df.rating)
mode = df.rating.mode().values[0]
sns.boxplot(data=df,x='rating',ax=ax_box)
ax_box.axvline(mean,color='r',linestyle='-')
ax_box.axvline(median,color='g',linestyle='-')
ax_box.axvline(mode,color='b',linestyle='-')
# plt.show()

sns.histplot(data=df,x='rating',ax=ax_hist,kde=True)
ax_hist.axvline(mean,color='r',linestyle='-',label='mean')
ax_hist.axvline(median,color='g',linestyle='-',label='median')
ax_hist.axvline(mode,color='y',linestyle='-',label='mode')
ax_hist.legend()
ax_box.set(xlabel='')
# plt.show()

df.rating.var()
df.rating.std()
mean = df.groupby(['movie'])['rating'].mean()
std = df.groupby(['movie'])['rating'].std()
fig,ax = plt.subplots()
mean.plot.bar(yerr=std,ax=ax,capsize=4)
# plt.show()

# Measures of spread
df1 = pd.DataFrame({'pop_sample':range(20)})

df1.sample(5).mean()
df1.sample(10).mean()
df1.mean()
import pandas as pd
from scipy import stats
stats.sem(df1)

df2= sns.load_dataset('tips')
sns.set_theme(style='whitegrid')
ax=sns.boxplot(x='day',y='total_bill',data=df2)
# plt.show()
ax=sns.boxplot(x='day',y='total_bill',data=df2)
ax=sns.swarmplot(x='day',y='total_bill',data=df2,color='0.25')
# plt.show()

print(df2['total_bill'].quantile([0.05,0.25,0.5,0.75]))
print(df2['total_bill'].quantile(0.75)-df2['total_bill'].quantile(0.25))

# Correlation and Covariance
df3= sns.load_dataset('iris')
numeric_df3 = df3.select_dtypes(include=['float64', 'int64'])
fig,ax=plt.subplots(figsize=(6,6))
ax = sns.heatmap(numeric_df3.corr(), vmin=-1, vmax=1, cmap=sns.diverging_palette(20, 220, as_cmap=True), ax=ax)
plt.tight_layout()
# plt.show()

# Covariance
a=[11,12,22,11]
b=[7,8,9,10]
c=[10,11,22,23]
arr=np.array([a,b,c])
cov_matrix=np.cov(arr,bias=True)
print(cov_matrix)
sns.heatmap(cov_matrix,annot=True,fmt='g')
# plt.show()

# Distribution

df = pd.DataFrame({
    'movie': ['a','b','c','a','b','c','a','b','c'],
    'rating': [9,7,6,9,5,7,8,9,5]
})
df_numeric = df.select_dtypes(include=[np.number])
df_numeric.skew()
df_numeric.kurtosis()
norm1 = np.arange(-5,5,0.001)
mean = 0.0
std = 1.0
pdf = stats.norm.pdf(norm1,mean,std)
plt.plot(norm1,pdf)
# plt.show()
import pylab
stats.probplot(df3.sepal_length,plot =pylab)
sns.kdeplot(df3.sepal_length)

from scipy.stats import binom
import matplotlib.pyplot as plt

n = 6
p = 0.5
r_value = list(range(n + 1))
dist = [binom.pmf(r, n, p) for r in r_value]

plt.bar(r_value, dist)
plt.show()
s= np.random.poisson(5,10000)
count,bins,ignored=plt.hist(s,10,density=True)
plt.show()

# CLT & confidence interval
import statsmodels.stats.api as sms
sms.DescrStatsW(df3.sepal_length).tconfint_mean()
fig,ax=plt.subplots()
ax2=ax.twinx()
n,bins,patches= ax.hist(df3.sepal_length,bins =100)
n,bins,patches= ax2.hist(df3.sepal_length,cumulative = 1,histtype='step',bins =100,color ='r')
plt.hist(df3.sepal_length,cumulative = True,label ='CDF',histtype ='step',alpha=0.8,color='y')

cdf= stats.norm.cdf(norm1)
plt.plot(norm1,cdf)
plt.show()

ax= sns.displot(df3.sepal_length)







