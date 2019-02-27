import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import  apriori
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

def get_freq(item):
    res=[]
    myRelated=freq_data[freq_data.itemsets.apply(lambda x : item in x)]
    list_val=list(myRelated.sort_values(by="support",ascending=False)['itemsets'])
    for sublist in list_val:
        for i in sublist:
            res.append(i)
    return set(res)

def get_freq_max(item):
    res=[]
    myRelated=freq_data[freq_data.itemsets.apply(lambda x : item in x)]
    list_val=list(myRelated[myRelated["len"] == myRelated["len"].max()].sort_values(by="support",ascending=False)['itemsets'])
    for sublist in list_val:
        for i in sublist:
            res.append(i)
    return set(res)

def get_ferq_with_txt(txt,tags):
    res=tags
    freq=[]
    for item in tags:
        freq.extend(list(get_freq_max(item)))
    for i in range(len(freq)):
        if freq[i] in txt:
            res.append(freq[i])
    if len(set(res)) == 0 :
        return(set(freq)) 
    elif len(set(res)) < 15:
        for j in range(len(freq)):
            if len(set(res)) < 15:
                res.append(freq[j])
            else:
                return set(res)
        return set(res)
    else:
        return set(res)

df=pd.read_csv("sample.csv")['Tags']
df=df.apply(lambda x:x.split(" "))
data=df.values
td= TransactionEncoder()
td_data=td.fit_transform(df)
df2=pd.DataFrame(td_data,columns=td.columns_)

freq_data = apriori(df2,min_support=0.0009,use_colnames=True)  #0.0004
freq_data['len']=freq_data.itemsets.apply(lambda x: len(x))

def test():
    df = pd.read_csv("sample.csv").head(1);
    txt=df["Title"]+" "+df["Body"]
    return get_ferq_with_txt(txt,["linux","c#","php"])
