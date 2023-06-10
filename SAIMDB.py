import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

##Reading the Dataset and reducing it to 10000 rows with 9000 being positive and 1000 negative


df_review = pd.read_csv('IMDB Dataset.csv')
df_review

df_positive = df_review[df_review['sentiment'] == 'positive'][:9000]
df_negative = df_review[df_review['sentiment'] == 'negative'][:1000]
df_review_imb = pd.concat([df_positive, df_negative])

##random sampling
rus = RandomUnderSampler(random_state=0)
df_review_bal, df_review_bal['sentiment'] = rus.fit_resample(df_review_imb[['review']], df_review_imb['sentiment'])

df_review_bal

##Comparison between balanced and imbalanced dataset
print(df_review_imb.value_counts('sentiment'))
print(df_review_bal.value_counts('sentiment'))

##splitting into train and test set
train, test = train_test_split(df_review_bal, test_size=0.33, random_state=42)

##set the independent and dependent variables
train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']

## turning the text data into numerical data
tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)
train_x_vector

# transform the test set
test_x_vector = tfidf.transform(test_x)

# Model Selection - since this is a binary classification (pos/neg) we will use classification algorithms

# SVM
from sklearn.svm import SVC

svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)

# test the SVC
print(svc.predict(tfidf.transform(['A good movie'])))
print(svc.predict(tfidf.transform(['An excellent movie'])))
print(svc.predict(tfidf.transform(['I did not like this movie at all'])))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x_vector, train_y)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(train_x_vector, train_y)

#svc.score('Test samples', 'True labels')

print(svc.score(test_x_vector, test_y))
print(dec_tree.score(test_x_vector, test_y))
print(gnb.score(test_x_vector.toarray(), test_y))
print(log_reg.score(test_x_vector, test_y))

#since SVM had the highest accuracy of 0.84 we shall focus on that for now

#F1 Score
from sklearn.metrics import f1_score
f1_score(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'], average=None)

from sklearn.metrics import classification_report
classification_report(test_y, svc.predict(test_x_vector), labels=['positive', 'negative'])

#Confusion Matrix

from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(test_y,
                            svc.predict(test_x_vector),
                            labels=['positive', 'negative'])

#Use GridSearchCV for optimization

from sklearn.model_selection import GridSearchCV
parameters = {'C': [1,4,8,16,32], 'kernel':['linear', 'rbf']}
svc = SVC()
svc_grid = GridSearchCV(svc,parameters, cv=5)
svc_grid.fit(train_x_vector, train_y)

print(svc_grid.best_params_)
print(svc_grid.best_estimator_)