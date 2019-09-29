"""@author:Izadi """

#import libraries
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
import os 
os.chdir(r'D:\desktop\Python_DM_ML_BA\sentiment_LAB ML')
os.getcwd()
#import the dataset
df = pd.read_csv("Restaurant_Reviews.tsv", sep='\t')
df.shape
df.columns
df.head()
df.isnull().sum()                                                        
df.info()
df.describe()
# To see the percentages of 0 and 1 which we see that they are equal.
d = df["Liked"].value_counts()
d
d.plot(kind = 'pie', figsize = (8, 8))
plt.ylabel("Liked vs Disliked")
plt.legend(["Liked", "Disliked"])
plt.show()


#filter punctuations, stemming and stopwords
corpus = []
for i in range(len(df)):
    review = re.sub('[^a-zA-Z0-9]', ' ', df['Review'][i])
    review = review.lower()
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    stemmer = nltk.PorterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()
    tokens_list = tokenizer.tokenize(review)
    tokens = []
    for token in tokens_list:
        tokens.append(stemmer.stem(lemmatizer.lemmatize(token)))
        stop_words = stopwords.words("english")
        filtered_words = [w for w in tokens if w not in stop_words]
        review = ' '.join(filtered_words)
        corpus.append(review)

len(corpus)
#Bag of Words model to convert corpus into X
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = pd.DataFrame(cv.fit_transform(corpus).toarray())
X.shape
y = df.Liked
y.shape
''' WE see that X has shape (11116, 1594) and y has shape (1000,)
the shape of DataFrame. To use the train_test_split they have to be consistent.
that is to say we have to choose only 1000 from the X.  '''
Z = X.head(1000)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Z,y,random_state=0)
#import classifier
from sklearn.metrics import accuracy_score,roc_curve ,confusion_matrix
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X_train, y_train)
y_pred = lg.predict(X_test)
accuracy_score(y_test, y_pred )
lg.score(X_test,y_test)
confusion_matrix(y_test,y_pred)
weights = lg.coef_
weights
intercept = lg.intercept_
intercept
'''we see that the accuracy is vey low. '''
#TF_IDF model to convert corpus into X
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X2 = pd.DataFrame(tfidf.fit_transform(corpus).toarray())
X2.shape # Agani the same shape as X.
Z = X2.tail(1000)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Z,y,random_state=0)
#import classifier
from sklearn.metrics import accuracy_score,roc_curve ,confusion_matrix
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X_train, y_train)
y_pred = lg.predict(X_test)
accuracy_score(y_test, y_pred )
lg.score(X_test,y_test)
confusion_matrix(y_test,y_pred)
weights = lg.coef_
weights
intercept = lg.intercept_
intercept
''''Again the samr accuracy. We have to do more work on the data'''


MostLiked = df.groupby("Review")["Liked"].agg([len, np.max]).sort_values(by = "len", ascending = False)
MostLiked.head(n=8)
'''Hence 3 dislikes are on the top and only one like with same len=2.
Lets study individual Liked/Disliked. '''

like = df[df["Liked"] == 1]["Review"]
dislike = df[df["Liked"] == 0]["Review"]
like_words = []
dislike_words = []
##################################################
#side work about isalpha()
str = "this"; # No space & digit in this string
str.isalpha()
str = "this is";
str.isalpha()
#####################################################
#importing important Tokenizers
from nltk.tokenize import WhitespaceTokenizer, WordPunctTokenizer, TreebankWordTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
stemmer = PorterStemmer()
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

def LikeWords(like):
    global like_words
    words = [word.lower() for word in word_tokenize(like) if word.lower() not in stopwords.words("english") and word.lower().isalpha()]
    like_words = like_words + words
    
def DislikeWords(dislike):
    global dislike_words
    words = [word.lower() for word in word_tokenize(dislike) if word.lower() not in stopwords.words("english") and word.lower().isalpha()]
    dislike_words = dislike_words + words

like.apply(LikeWords)
dislike.apply(DislikeWords)
from wordcloud import WordCloud
#Spam Word cloud
like_wordcloud = WordCloud(width=600, height=400).generate(" ".join(like_words))

plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(like_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

dislike_wordcloud = WordCloud(width=600, height=400).generate(" ".join(dislike_words))

plt.figure( figsize=(10,8), facecolor='m')
plt.imshow(dislike_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
# Top 20 spam words
like_words = np.array(like_words)

print("Top 8 like_words are :\n")
pd.Series(like_words).value_counts().head(n = 8)
# Top 20 Ham words
dislike_words = np.array(dislike_words)

print("Top 10 dislike_words are :\n")
pd.Series(dislike_words).value_counts().head(n = 8)
#Does the length of the message indicates us anything?¶
df["reviewLength"] = df["Review"].apply(len)
df["reviewLength"].describe()

'''TEXT TRANSFORMATION
#Lets remove punctuations/ stopwords and stemming words'''
import string
string.punctuation
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")

def cleanText(review):
    review = review.translate(str.maketrans('', '', string.punctuation))
    words = [stemmer.stem(word) for word in review.split() if word.lower() not in stopwords.words("english")]
    return " ".join(words)

df["Review"] = df["Review"].apply(cleanText)
df.shape
df.head(n = 12) 
'''Machine learning model using Tfifd '''
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")
features = vec.fit_transform(df["Review"])
features.shape
X = features 
y = df.Liked
# train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
from sklearn.metrics import accuracy_score,confusion_matrix
#importing RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier 
rd=RandomForestClassifier()
rd.fit(X_train,y_train)
y_pred= rd.predict(X_test)
accuracy_score(y_pred,y_test)
rd.score(X_test,y_test)
confusion_matrix(y_test,y_pred)
# We see some impovment

from sklearn.metrics import roc_curve, auc,roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
roc_auc

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# Let us use GridSearchCV with 10 cv
from sklearn.model_selection import GridSearchCV

pram_dict ={'max_depth':[75,80,85],
           'n_estimators':[35,40,45]}
           
                                   
gscv =  GridSearchCV(rd, pram_dict,cv=10)                              
gscv.fit(X_train, y_train) 
gscv.score(X_test,y_test)
y_pred= gscv.predict(X_test)
accuracy_score(y_pred,y_test)
confusion_matrix(y_test,y_pred)
# Finally MultinomialNB
from sklearn.metrics import fbeta_score
from sklearn.naive_bayes import MultinomialNB
gaussianNb = MultinomialNB()
gaussianNb.fit(X_train, y_train)
y_pred = gaussianNb.predict(X_test)
fbeta_score(y_test, y_pred, beta = 0.5)
#Agani an improvment of %10.




























