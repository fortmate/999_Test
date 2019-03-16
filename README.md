#0.75になるという触れこみ→結果：０.５。メクラ判定で範囲指定で生死判定出してもそれくらい行きそうだ。
#原因分析と、改善する必要あり。
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv as csv
%matplotlib inline

#Format input
df = pd.read_csv("../input/train.csv")
#1 df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df.Embarked = df.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])
df.Sex = df.Sex.replace(['male', 'female'], [0, 1])
df.Age = df.Age.replace('NaN', 0)
#Learn
#corrmat = df.corr()
#f, ax = plt.subplots(figsize=(12,9))
#sns.heatmap(corrmat, vmax=.8, square=True)

train_labels = df['Survived'].values
train_features = df
train_features.drop('Survived', axis=1, inplace=True)
train_features = train_features.values.astype(np.int64)
from sklearn import svm
#Standard = svm.LinearSVC(C=1.0, intercept_scaling=1, multi_class=False , loss="l1", penalty="l2", dual=True)
svm = svm.LinearSVC(dual=False)
svm.fit(train_features, train_labels)

#test
df_test = pd.read_csv("../input/test.csv")
df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_test.Embarked = df_test.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])
#df.Cabin = df.Cabin.replace('NaN', 0)
df_test.Sex = df_test.Sex.replace(['male', 'female'], [0, 1])
df_test.Age = df_test.Age.replace('NaN', 0)

test_features = df_test.values.astype(np.int64)
y_test_pred = svm.predict(test_features)

df_out = pd.read_csv("../input/test.csv")
df_out["Survived"] = y_test_pred
#df_out[["PassengerId","Survived"]].to_csv("../submission.csv",index=False)
ids = df_out["PassengerId"].values
svs = df_out["Survived"].values

# export result to be "titanic_submit.csv"
submit_file = open("titanic_submit.csv", "w")
file_object = csv.writer(submit_file)
file_object.writerow(["PassengerId", "Survived"])
file_object.writerows(zip(ids,svs))
submit_file.close()
