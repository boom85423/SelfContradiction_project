import itertools
import numpy as np
import pickle
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import pyprind
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
from utils import Evaluate
from ISIF import Normalize, Criterion_aug


with open('./dataset/corpus_clear_splitedByTitleBalanced.pickle', 'rb') as file:
    corpus = pickle.load(file)
    train_X_title = corpus["train_X_title"]
    train_X_raw = corpus["train_X_raw"]
    train_Y = corpus["train_Y"]
    train_revision_id = corpus['train_revision_id']
    test_X_title = corpus["test_X_title"]
    test_X_raw = corpus["test_X_raw"]
    test_Y = corpus["test_Y"]
    test_revision_id = corpus['test_revision_id']


class XL_Dataset(torch.utils.data.Dataset):
    def __init__(self, sentences, labels, tokenizer):
          self.sentences = sentences
          self.labels = labels
          self.tokenizer = tokenizer
          self.len = len(labels)

    def __len__(self):
          return self.len

    def __getitem__(self, idx):
         sentences = self.sentences[idx]
         labels = self.labels[idx]
         
         word_pieces = ["[CLS]"]
         segments = []
         counter = 0
         for i in range(len(sentences)):
             tokens = self.tokenizer.tokenize(sentences[i])
             word_pieces += tokens + ["[SEP]"]
             segments.extend([i] * (len(word_pieces)-len(segments)))
         ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
         tokens_tensor = torch.tensor(ids)
         segments_tensor = torch.tensor(segments)
         
         labels_tensor = torch.tensor(labels)
         return (tokens_tensor, segments_tensor, labels_tensor)

def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)

    segments_tensors = [s[1] for s in samples]
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)

    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    
    labels_tensors = torch.stack([s[2] for s in samples])
    return tokens_tensors, segments_tensors, masks_tensors, labels_tensors


class LSTM_classifier(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, num_class):
        super(LSTM_classifier, self).__init__()
        self.vocab_size, self.input_dim, self.hidden_dim, self.num_class = vocab_size, input_dim, hidden_dim, num_class
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        # self.attention_weight_branch = nn.Linear(hidden_dim, 1)
        self.attention_weight = nn.Linear(hidden_dim*2, 1)
        self.fc = nn.Linear(hidden_dim*2, num_class)

    # def Attention_Layer_branch(self,hidden_vec):
    #     attn = self.attention_weight_branch(hidden_vec)
    #     attn = F.softmax(attn, 0)   
    #     out = torch.sum(hidden_vec * attn, 0)
    #     return out

    def Attention_Layer(self,hidden_vec):
        attn = self.attention_weight(hidden_vec)
        attn = F.softmax(attn, 0)
        out = torch.sum(hidden_vec * attn, 0)
        return out

    def forward(self, X, masks_tensors):
        X = self.embedding(X)
        hidden_vec, (h_n, c_n) = self.lstm(X.permute(1, 0, 2))
        
        features = torch.cat([h_n[0], h_n[1]], 0)
        # hidden_vec1 = hidden_vec[:, :, :self.hidden_dim] 
        # hidden_vec2 = hidden_vec[:, :, self.hidden_dim:]
        # features1 = self.Attention_Layer_branch(hidden_vec1)
        # features2 = self.Attention_Layer_branch(hidden_vec2)
        # features = torch.cat([features1, features2], 0)

        attn_layer_out = self.Attention_Layer(hidden_vec)
        out = self.fc(attn_layer_out)
        return features, F.softmax(out, 1)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

trainset = XL_Dataset(train_X_raw, train_Y, tokenizer)
trainloader = DataLoader(trainset, batch_size=32, collate_fn=create_mini_batch, shuffle=True)
testset = XL_Dataset(test_X_raw, test_Y, tokenizer)
testloader = DataLoader(testset, batch_size=32, collate_fn=create_mini_batch, shuffle=True)

model = LSTM_classifier(30522, 128, 128, 2)
model.to(device)

criterion_cls = nn.CrossEntropyLoss()
criterion_cls.to(device)
criterion_aug = Criterion_aug(1, 0.1, 32)
criterion_aug.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
norm = Normalize()

for epoch in range(10):
    ## training
    for i, data in enumerate(trainloader):
        optimizer.zero_grad()

        tokens_tensors, segments_tensors, masks_tensors, labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
        features, outputs = model(tokens_tensors, masks_tensors)

        loss_cls = criterion_cls(outputs, labels)
        loss_aug = criterion_aug(norm(features))
        if i % 20 == 0:
            print("loss_cls: %.4f,  loss_aug: %.4f" % (loss_cls, loss_aug))
        loss = loss_cls + loss_aug
        loss.backward()
        optimizer.step()

    ## evaluation
    correct = 0
    total = 0
    pred = []
    true = []
    with torch.no_grad():
        for i, data in enumerate(testloader):
            tokens_tensors, segments_tensors, masks_tensors, labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
            _, outputs = model(tokens_tensors, masks_tensors)
            _, predicted = torch.max(outputs.data, 1)
            pred.extend(predicted.cpu().tolist())
            true.extend(labels.cpu().tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("accuracy: %.4f\n" % (correct /total))
Evaluate(true, pred)

