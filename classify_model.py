import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.optim import optimizer
import transformers
from transformers import AutoModel, AutoTokenizer
from keras.preprocessing.sequence import pad_sequences
import json
#from vncorenlp import VnCoreNLP
from sklearn.utils import shuffle
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from underthesea import word_tokenize

#Đọc dữ liệu
def get_data(all_path):
    sentences=[]
    labels=[]
    for i in all_path:
        try:
            with open(i, "r", encoding='utf-8') as f:
                datastore = json.load(f)
        except FileNotFoundError:
            print(f"Error: File {i} not found")
            continue
        except json.JSONDecodeError:
            print(f"Error: File {i} contains invalid JSON")
            continue
        for item in datastore:
            sentences.append(item["sentences"])
            labels.append(item["toxic"])
    return sentences, labels


#Tách từ tiếng việt
# rdrsegmenter=VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
# def sentences_segment(sentences):
#     for i in range(len(sentences)):
#         tokens=rdrsegmenter.tokenize(sentences[i])
#         statement=""
#         for token in tokens:
#             statement+=" ".join(token)
#         sentences[i]=statement
def sentences_segment(sentences):
    for i in range(len(sentences)):
        # word_tokenize trả về chuỗi các từ cách nhau bằng dấu space
        sentences[i] = word_tokenize(sentences[i], format="text")

#Mã hóa các câu thành Token ID và pad chuỗi về độ dài maxlen
phobert=AutoModel.from_pretrained('vinai/phobert-base')
tokenizer=AutoTokenizer.from_pretrained('vinai/phobert-base')
def shuffle_and_tokenize(sentences,labels,maxlen):
    sentences,labels=shuffle(sentences,labels)
    sequences=[tokenizer.encode(i) for i in sentences]
    labels=[int(i) for i in labels]
    padded=pad_sequences(sequences, maxlen=maxlen, padding="pre")
    return padded, labels

def check_maxlen(sentences):
    sentences_len=[len(i.split()) for i in sentences]
    return max(sentences_len)


#Chia dữ liệu huấn luyện/val/test
def split_data(padded, labels):
    padded=torch.tensor(padded)
    labels=torch.tensor(labels)
    X_train,X_,y_train,y_=train_test_split(padded, labels,random_state=2018, train_size=0.8, stratify=labels)
    X_val,X_test, y_val, y_test=train_test_split(X_, y_, random_state=2018, train_size=0.5, stratify=y_)
    return X_train,X_val,X_test, y_train,y_val, y_test


#Tạo DataLoader
def Data_Loader(X_train,X_val,y_train,y_val):
    train_data=TensorDataset(X_train,y_train)
    train_sampler=RandomSampler(train_data)
    train_dataloader=DataLoader(train_data, sampler=train_sampler,batch_size=2)
    val_data=TensorDataset(X_val,y_val)
    val_sampler=RandomSampler(val_data)
    val_dataloader=DataLoader(val_data, sampler=val_sampler,batch_size=2)
    return train_dataloader, val_dataloader

# Chuẩn bị dữ liệu (chỉ chạy 1 lần ở train_model.py)
sentences,labels=get_data(['toxic_dataset.json','normal_dataset.json'])
sentences_segment(sentences)
padded,labels=shuffle_and_tokenize(sentences,labels,check_maxlen(sentences))
X_train,X_val,X_test, y_train,y_val, y_test=split_data(padded, labels)
train_dataloader, val_dataloader=Data_Loader(X_train,X_val,y_train,y_val)

# Freeze PhoBERT để không train lại
for param in phobert.parameters():
    param.requires_grad=False

class classify(nn.Module):
    def __init__(self, phobert, number_of_category):
        super(classify,self).__init__()
        self.phobert=phobert
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.1)
        self.first_function=nn.Linear(768, 512)
        self.second_function=nn.Linear(512, 32)
        self.third_function=nn.Linear(32,number_of_category)
        self.softmax=nn.LogSoftmax(dim=1)

    def forward(self, input):
        x=self.phobert(input)
        x=self.first_function(x[1])
        x=self.relu(x)
        x=self.dropout(x)
        x=self.second_function(x)
        x=self.relu(x)
        x=self.third_function(x)
        x=self.softmax(x)
        return x

cross_entropy=nn.NLLLoss()
model=classify(phobert,2)
optimizer=AdamW(model.parameters(),lr=1e-5)

#Train / Evaluate
def train():
    model.train()
    total_loss=0
    total_preds=[]
    for step , batch in enumerate(train_dataloader):
        if step%50==0 and step!=0:
            print("BATCH {} of {}".format(step, len(train_dataloader)))
        
        input,labels=batch
        model.zero_grad()
        preds=model(input)
        loss=cross_entropy(preds, labels)
        total_loss+=loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds=preds.detach().cpu().numpy()
        total_preds.append(preds)
    avg_loss=total_loss/len(train_dataloader)
    total_preds=np.concatenate(total_preds,axis=0)
    return avg_loss, total_preds

def evaluate():
    model.eval()
    total_loss=0
    total_preds=[]
    for step, batch in enumerate(val_dataloader):
        if step%50==0 and step!=0:
            print("BATCH {} of {}".format(step, len(val_dataloader)))
        input,labels=batch
        with torch.no_grad():
            preds=model(input)
            loss=cross_entropy(preds, labels)
            total_loss+=loss.item()
            preds=preds.detach().cpu().numpy()
            total_preds.append(preds)
    avg_loss=total_loss/len(val_dataloader)
    total_preds=np.concatenate(total_preds,axis=0)
    return avg_loss, total_preds


#Huấn luyện toàn bộ
def run(epochs):
    best_valid_loss=float("inf")
    train_losses=[]
    valid_losses=[]
    for epoch in range(epochs):
        print("EPOCH {}/{}".format(epoch+1, epochs))
        train_loss,_ =train()
        valid_loss,_ =evaluate()
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(),"save_weights.pt")
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f"Train Loss: {train_loss}, Val Loss: {valid_loss}")
        #print(f"Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}")
        # print(train_loss)
        # print(valid_loss)

if __name__ == "__main__":
    print("Module classify_model.py đã được load. Không chạy trực tiếp.")
