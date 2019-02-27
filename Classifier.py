import pandas as pd
import nltk
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
stop_words = set(stopwords.words('english'))


def clean(text):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    text = re.sub(clean, ' ', text)
    text = re.sub('[^A-Za-z #]+', ' ', text)
    #remove space one letter space
    text = re.sub('(\s\w{1}\s)+(\w{1}\s)*',' ', text)
    #remove space tow letter space
    text = re.sub('(\s\w{2}\s)+(\w{2}\s)*',' ', text)
    #remove space # space
    text = re.sub('(\s\W{1}\s)+(\W{1}\s)*',' ',text)
    return str(text).lower()

def stemm_stop(text):
    ps  = PorterStemmer()
    #stop_words = stopwords.words("english")
    stopwords = nltk.corpus.stopwords.words('english')
    newStopWords = ['num','na','#']
    stopwords.extend(newStopWords) 
    filtered_words = []
    for i in text.split():
        if i not in stopwords:
            filtered_words.append(ps.stem(i))
    return " ".join(filtered_words)


def tokenzer(text):
    tk = TweetTokenizer()
    return tk.tokenize(text)


df = pd.read_csv("sample.csv").head(1000);
df["Text"] = df['Body']+' ' + df['Title']
df_new=df[["Text","Tags"]]
df_new['Text'] = df_new['Text'].apply(clean)
df_new['Text'] = df_new['Text'].apply(stemm_stop)

my_list=[]
for i in range(len(df_new)):
    splitedData=df_new['Tags'][i].split(" ")
    if(len(splitedData)):
        for j in range(len(splitedData)):
            my_list.append([df_new['Text'][i],splitedData[j]])
    else:
          my_list.append([df_new['Text'][i],df_new['Text'][i]])

D_F=pd.DataFrame(my_list,columns=["Text","Tags"])
TFIDF = TfidfVectorizer(tokenizer=tokenzer,analyzer="word",max_features=10000)
x_Data   = TFIDF.fit_transform(D_F['Text']).toarray()

D_W=pd.DataFrame(x_Data,columns=TFIDF.get_feature_names())

x_train,x_test,y_train,y_test = train_test_split(D_W,D_F["Tags"],test_size=0.3)

lr=LogisticRegression()
lr.fit(x_train,y_train)

def get_score():
    return lr.score(x_test,y_test)