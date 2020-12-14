import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
sns.set()
games = pd.read_csv("games.csv")
print(games.dtypes)
games['winner']=games['winner'].astype('category')
games['winnerEnc']=games['winner'].cat.codes
games['rated']=games['rated'].astype('category')
games['victory_status']=games['victory_status'].astype('category')
games['ratedEnc']=games['rated'].cat.codes
games['statusEnc']=games['victory_status'].cat.codes
games = (
    games.assign(
        opening_archetype=games.opening_name.map(
            lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()
        ),
        opening_moves=games.apply(lambda srs: srs['moves'].split(" ")[:srs['opening_ply']],
                                  axis=1)
    )
)
games['opening_archetype']=games['opening_archetype'].astype('category')
games['archeEnc']=games['opening_archetype'].cat.codes

X=games[['archeEnc','statusEnc','white_rating','black_rating']]
y=games['winnerEnc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)

#Logistic Regression

from sklearn import linear_model

reg = linear_model.LogisticRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# making predictions on the testing set
y_pred = reg.predict(X_test)

# comparing actual response values (y_test) with predicted response values (y_pred)

print("Confusion Matrix\n",metrics.confusion_matrix(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) #65.71

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# making predictions on the testing set
y_pred = gnb.predict(X_test)

# comparing actual response values (y_test) with predicted response values (y_pred)

print("Confusion Matrix\n",metrics.confusion_matrix(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))   #65.76

#KNN

X=games[['archeEnc','statusEnc','turns']]
y=games['winnerEnc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=109) # 70% training and 30% test
knn = KNeighborsClassifier(n_neighbors=1)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)
print("Confusion Matrix\n",confusion_matrix(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))   #68.94

#Random Forest
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)
from sklearn import model_selection
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
# predictions
y_pred = rfc.predict(X_test)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

print("Confusion Matrix\n",confusion_matrix(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))  #73.52

#SVM
X=games[['archeEnc','statusEnc','turns','white_rating','black_rating']]
y=games['winnerEnc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=109)
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) #65.9%  for full data 55%


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Confusion Matrix\n",metrics.confusion_matrix(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))  #86.15

#To predict
le = preprocessing.LabelEncoder()
winner_encoded=le.fit_transform(games.winner)  #white-2,draw-1,black-0
#print(winner_encoded)
rated_encoded=le.fit_transform(games.rated)    #0-false,1-true
#print(rated_encoded)
status=le.fit_transform(games.victory_status)    #2-Outoftime,3-resign,1-mate,0-draw
#print(status)
features=list(zip(rated_encoded,status))
model = KNeighborsClassifier(n_neighbors=3)
model.fit(features,winner_encoded)
predicted= model.predict([[0,0]]) # 1:True, 0:draw
print(predicted)    #draw

#**********************************************************************************************
games_played = pd.concat([games['white_id'], games['black_id']]).value_counts()
#print(games_played)
games_played.reset_index(drop=True).plot.line(figsize=(16, 8), fontsize=18)

n_ge_2 = len(games_played[games_played > 1])
print(str(n_ge_2) + " players who have played at least two games.")


plt.axvline(n_ge_2, color='green')
plt.show()

len(games_played)
games_played[games_played > 1].sum()
#print(games_played)

opening_used = (pd.concat([
                   games.groupby('white_id')['opening_archetype'].value_counts(),
                   games.groupby('black_id')['opening_archetype'].value_counts()
                ])
                    .rename(index='openings_used')
                    .reset_index()
                    .rename(columns={'white_id': 'player_id', 'openings_used': 'times_used'})
                    .groupby(['player_id', 'opening_archetype']).sum()
               )
#print(opening_used.head(10))
print(opening_used
     .reset_index()
     .groupby('player_id')
     .filter(lambda df: df.opening_archetype.isin(["Queen's Gambit Accepted"]).any())
     .query('opening_archetype != "Queen\'s Gambit Accepted"')
     .groupby('opening_archetype')
     .times_used
     .sum()
     .sort_values(ascending=False)
     .to_frame()
     .pipe(lambda df: df.assign(times_used = df.times_used / df.times_used.sum()))
     .squeeze()
     .head(10)
)
print(opening_used
     .reset_index()
     .groupby('player_id')
     .filter(lambda df: df.opening_archetype.isin(["Italian Game"]).any())
     .query('opening_archetype != "Italian Game"')
     .groupby('opening_archetype')
     .times_used
     .sum()
     .sort_values(ascending=False)
     .to_frame()
     .pipe(lambda df: df.assign(times_used = df.times_used / df.times_used.sum()))
     .squeeze()
     .head(10)
)
#Recommending moves
def threshold_map(n_opening, n_all):
    if pd.isnull(n_opening):
        return np.nan
    elif n_opening / n_all >= 1 / 4:
        return 5
    elif n_opening / n_all >= 1 / 8:
        return 4
    elif n_opening / n_all > 1 / 16:
        return 3
    else:
        return 2

recommendations = opening_used.unstack(-1).loc[:, 'times_used'].apply(
    lambda srs: srs.map(lambda v: threshold_map(v, srs.sum())),
    axis='columns'
)
#print(recommendations.head())
#pd.Series(recommendations.values.flatten()).value_counts().sort_index().plot.bar()
#plt.show()
from sklearn.metrics.pairwise import pairwise_distances
# user_similarity = pairwise_distances(train.fillna(0), metric='cosine')
item_similarity = pairwise_distances(recommendations.T.fillna(0), metric='cosine')
print(item_similarity.shape)
correction = np.array([np.abs(item_similarity).sum(axis=1)])
item_predictions = recommendations.fillna(0).dot(item_similarity).apply(
    lambda srs: srs / np.array([item_similarity.sum(axis=1)]).flatten(), axis='columns')
print(item_predictions.head())
recommended_opening_numbers = item_predictions.apply(
    lambda srs: np.argmax(srs.values), axis='columns'
)
recommended_opening_numbers.head()
opening_names = pd.Series(recommendations.columns)
recommended_openings = recommended_opening_numbers.map(opening_names)
recommended_openings.head()
recommended_openings.value_counts().head(10).iloc[1:].plot.bar(figsize=(24, 10), fontsize=10)
plt.xticks(rotation=15)
plt.xlabel("Opening Name")
plt.ylabel("Count")
plt.title("Recommended Moves")
plt.show()
#create a accuracy dataframe
data = [['logistic',0.657195],['KNN',0.689431],['Decision Tree',0.861581],['Random Forest',0.735294],['SVM',0.659],['Naive Bayes',0.657652]]
df = pd.DataFrame(data,columns=['ALGORITHM','ACCURACY'],index=[1,2,3,4,5,6])
print(df)

#*********************************************************************

#Scatter Plot
plt.plot(games['victory_status'],games['winner'],'o')
plt.style.use('ggplot')
plt.xlabel("Victory Status")
plt.ylabel("Winner")
plt.title("Scatter Plot")
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

#jointplot
sns.set(style="white")
sns.jointplot(games.white_rating, games.black_rating, kind="kde", height=7, space=0)

plt.show()

#cat plot
pl = sns.catplot(x='victory_status',y='turns',hue='winner',data=games,
                height=6, kind="bar", palette="muted")
pl.despine(left=True)
plt.show()

#Count plot
sns.countplot(x='winner',data=games)
plt.show()