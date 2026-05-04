import pandas as pd

df=pd.read_csv("IMDB Dataset.csv")
df.drop_duplicates(inplace=True)

#-----------preprocessing--------------
#conver to lowercase-->remove URL-->HTML Tags-->PUNCTUATIONS-->Stopword-->stemming-->encode sentiment-->vectorization

df["review"]=df["review"].str.lower()

import re
def remove_url(text):
    text=re.sub(r"http\s+","",text)
    return text
df["review"]=df["review"].apply(remove_url)

def remove_punctuation(text):
    text=re.sub(r"[^A-Za-z0-9\s]", "",text)
    return text
df["review"]=df["review"].apply(remove_punctuation)

def remove_html(text):
    text=re.sub(r"<.*?>","",text)
    return text
df["review"]=df["review"].apply(remove_html)

#tokanize
import nltk
nltk.download("punkt")
#nltk.download("punkt_tab")
nltk.download("stopwords")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def remove_stopwords(text):
    tokens=word_tokenize(text)
    stop_words=stopwords.words("english")

    for word in tokens:
        if word in stop_words:
            text=text.replace(word,"")
    return text
df["review"]=df["review"].apply(remove_stopwords)

#stemming   plyed-->play
from nltk.stem import PorterStemmer

def stemming(text):
    ps=PorterStemmer()
    stemmed_word=[]

    tokens=word_tokenize(text)
    for token in tokens:
        stemed_token=ps.stem(token)
        stemmed_word.append(stemed_token)
    return " ".join(stemmed_word)
df["review"]=df["review"].apply(stemming)

#------encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df["sentiment"]=le.fit_transform(df["sentiment"])
y=df["sentiment"]

#------vectorization---------
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer(max_features=5000)
x=tf.fit_transform(df["review"])

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from torch.utils.data import TensorDataset,DataLoader
import torch

x_train=x_train.toarray()
x_test=x_test.toarray()
train_set =TensorDataset(
    torch.from_numpy(x_train).float(),
    torch.from_numpy(y_train.values).float()
)
test_set =TensorDataset(
    torch.from_numpy(x_test).float(),
    torch.from_numpy(y_test.values).float()
)

train_loader=DataLoader(train_set,shuffle=True,batch_size=64)
test_loader=DataLoader(test_set,shuffle=True,batch_size=64)

import torch.nn as nn
import torch.optim as optim
class RNN(nn.Module):
    def __init__(self,input_size,hidden_size=128,num_layers=1):
        super().__init__()

        self.hidden_size=hidden_size
        self.num_layers=num_layers
        #RNN Layer
        self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)

        #fc layer
        self.fc=nn.Linear(hidden_size,1)

    def forward(self,x):
        #shape(num of layers,batch size,hidden size)
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        out,_=self.rnn(x,h0)
        out=self.fc(out[:,-1,:])
        return out

input_size=x_train.shape[1]
model=RNN(input_size)

criteron=nn.BCELoss()
optimzer=optim.Adam(model.parameters())

#training RNN
epochs=10
for epoch in range(epochs):
    model.train()
    for xb,yb in train_loader:
        optimzer.zero_grad()
        xb=xb.unsqueeze(1)#add single direction
        outputs=model(xb)
        outputs=torch.sigmoid(outputs.squeeze())
        loss=criteron(outputs,yb)
        loss.backward()
        optimzer.step()
    
    print(f"{epoch} and loss = {loss.item()}")

#evaluate

model.eval()
with torch.no_grad():
    correct_vals=0
    tot_vals=0

    for xb,yb in test_loader:
        xb=xb.unsqueeze(1)

        outputs=model(xb)
        predicted=(torch.sigmoid(outputs.squeeze())>0.5).float()
        tot_vals+=yb.size(0)
        correct_vals+=(predicted==yb).sum().item()
    print(f"Accuracy = {correct_vals/tot_vals*100}")

print(x)
