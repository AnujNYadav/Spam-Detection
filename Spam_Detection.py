

# Load data
import pandas as pd

data = pd.read_csv('SmsSpamCollection/SMSSpamCollection',sep='\t',names=['label','message'])
data.head()


l = data.shape
print(l)



# Text cleaning and preprocessing

# import libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()
corpus = []

# cleaning and preprocessing
for i in range(l[0]):
    words = re.sub('[^a-zA-Z]',' ',data['message'][i])              # Remove all character except alphabets
    words = words.lower()                                           # Lowercase words
    words = words.split()                                           # split the words
    words = [ps.stem(word) for word in words if not word in stopwords.words('english')]
    words = ' '.join(words)
    corpus.append(words) 



# apply bag of words model

from sklearn.feature_extraction.text import CountVectorizer
bag = CountVectorizer()
X = bag.fit_transform(corpus).toarray()
X.shape


# label target variable

Y=pd.get_dummies(data['label'])
Y=Y.iloc[:,1].values
print(Y)


# split data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

x_train.shape


# training data using naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
spam_detection_model = NB.fit(x_train,y_train)

# prediction

y_pred = spam_detection_model.predict(x_test


# Confusion matrix and Accuracy

from sklearn.metrics import confusion_matrix, accuracy_score
confusion_m = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print(confusion_m)
print(accuracy)



