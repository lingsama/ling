#Recommender System for Stock Trading and Electronic Trading Alerts Digital Assistant with friendly GUI

![Let's use this flow for the login experience](https://img-s-msn-com.akamaized.net/tenant/amp/entityid/AAwhzxJ.img?h=1142&w=728&m=6&q=60&o=f&l=f&x=613&y=242)
##Code
<code>
	# -*- coding: utf-8 -*-
	"""
	statistical machine learning algorithms for this problem
	http://scikit-learn.org
	"""
	# Import dependencies
	import seaborn as sns
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd
	
	# Read data
	df = pd.read_csv('Combined_News_DJIA1.csv', parse_dates=True, index_col=0)
	#print('read csv')
# Plot class distribution
sns.countplot(x='Label', data=df)
#????
#plt.show()

# Check the data
df.head()
#print(df)
#print('--------------------')
# Select only the top N news to use as features of the classifier. N ranges from 1 to 25.
# In this case, N = 20. 
columns = ['Top' + str(i+1) for i in range(20)]
#print (columns)

# Create a new column with the Top N news joined.
df['joined'] = df[columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)

# Create a new dataframe with only Label and joined columns
df1 = df[['Label', 'joined']].copy()

# Take a look
df1.head()
#print(df1)

# According to the author of the dataset, the data should be split as it:
# Train set: from 2008-08-08 to 2014-12-31 
# Test set: from 2015-01-02 to 2016-07-01
#print('---------------------')
train = df1.ix['2008-08-08':'2014-12-31'].shape
test = df1.ix['2015-01-02':'2016-07-01'].shape
#print('this is train:',train)
#test(378,2)
#print('this is test:',test)

from sklearn.feature_extraction.text import TfidfVectorizer
# Create a tfidf object. Remove english stop words and use 10000 features.
vect = TfidfVectorizer(max_features=10000, stop_words='english')
#print(vect)

# Transform the joined column into a tfidf sparse matrix
X = vect.fit_transform(df1['joined'])
#print(X)
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# Use tfidf followed by svd is known as lsa or lsi.
svd = TruncatedSVD(1000)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

# Apply lsa
X = lsa.fit_transform(X)
#print(X)

# Split data into train and test
X_train = X[:1611]
#print (X_train.shape)
X_test = X[1611:]
#print (X_test.shape)

# Get labels
y_train = df1['Label'].ix['2008-08-08':'2014-12-31']
y_test = df1['Label'].ix['2015-01-02':'2016-07-01']

#print(y_test)
#print(type(y_test))


from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model.logistic import LogisticRegression

# Create for different classifiers. Random Forest, KNN, Support Vector Machine and an Ensemble of these 3.
rf = RandomForestClassifier()
knn = KNeighborsClassifier(n_neighbors=3)
svm = LinearSVC()
vc = VotingClassifier(estimators=[('rf',rf), ('knn',knn), ('svm',svm)])
lf = LogisticRegression()

list_rf = []
list_knn = []
list_svm = []
list_vc = []
list_lf = []
list_i = []



for i in range(0,5):
print('this is',i,'time')
# Train them
rf.fit(X_train, y_train)
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)
vc.fit(X_train, y_train)
lf.fit(X_train, y_train)

# Check the accuracies
ac_rf = rf.score(X_test, y_test)
ac_knn = knn.score(X_test, y_test)
ac_svm = svm.score(X_test, y_test)
ac_vc = vc.score(X_test, y_test)
ac_lf = lf.score(X_test, y_test)

list_rf.append(ac_rf)
list_knn.append(ac_knn)
list_svm.append(ac_svm)
list_vc.append(ac_vc)
list_lf.append(ac_lf)
list_i.append(i+1)


print ("Accuracy rf is", ac_rf)
print ("Accuracy knn is", ac_knn)
print ("Accuracy svm is", ac_svm)
print ("Accuracy vc is", ac_vc)
print ("Accuracy lf is", ac_lf)
print('--------------------------')
print('END')

print('random forest:',list_rf)
print('KNN:',list_knn)
print('SVM:',list_svm)
print('VC:',list_vc)
print('LF:',list_lf)

	df_all = pd.DataFrame({"rf":list_rf,"knn":list_knn,"svm":list_svm,"vc":list_vc,"lf":list_lf,"i":list_i})
	print(df_all)
	sns.jointplot(x='i',y='rf',data=df_all)
	plt.show()

</code>