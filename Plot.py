import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import preprocessing
from sklearn import metrics
sns.set()
games = pd.read_csv("games.csv")
print(games.dtypes)
games['winner']=games['winner'].astype('category')
games['winnerEnc']=games['winner'].cat.codes
#print(games.head(10))
games['rated']=games['rated'].astype('category')
games['victory_status']=games['victory_status'].astype('category')
games['ratedEnc']=games['rated'].cat.codes
games['statusEnc']=games['victory_status'].cat.codes
#print(games.head(10))
plt.plot(games['victory_status'],games['winner'],'o')
plt.style.use('ggplot')
plt.xlabel("Victory Status")
plt.ylabel("Winner")
plt.title("Scatter Plot")

plt.show()

n = games.groupby('winner').count()
d = games.groupby('victory_status').count()
print(d['statusEnc'])
print(n['winnerEnc'])

black= games[games['winner']=='black']
white= games[games['winner']=='white']
print(black)
print(white)
w_group=white.groupby(['victory_status'])
b_group=black.groupby(['victory_status'])
coublack= games.loc[games['winnerEnc']==0].count()
coudraw = games.loc[games['winnerEnc']==1].count()
couwhite = games.loc[games['winnerEnc']==2].count()


#bar plot
pl = sns.catplot(x='victory_status',y='turns',hue='winner',data=games,
                height=6, kind="bar", palette="muted")
pl.despine(left=True)


plt.show()

#relplot
sns.set(style="darkgrid")
sns.relplot(x="opening_ply", y="statusEnc", hue="winner", style="rated",ci=None,
            dashes=False, markers=True, kind="line", data=games);
plt.ylabel("Victory Status")
plt.show()
black = games.query("winner == 'black'")
white = games.query("winner == 'white'")
draw = games.query("winner == 'draw'")

#kdeplot
sns.set(style="white")
sns.jointplot(games.white_rating, games.black_rating, kind="kde", height=7, space=0)

plt.show()
#corr matrix
corr = games.corr()
matrix = np.triu(corr)
sns.heatmap(corr,mask=matrix,annot=True,square=True)
plt.xticks(rotation=45)
plt.title("Correlation")
plt.show()
sns.countplot(x='winner',data=games)
plt.show()







