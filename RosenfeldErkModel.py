import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Model(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super(Model, self).__init__()
        self.emb_dimension = emb_dimension
        # time encoder
        self.dense1 = nn.Linear(emb_dimension,emb_dimension)
        self.dense2 = nn.Linear(emb_dimension, emb_dimension)
        self.dense4 = nn.Linear(emb_dimension, emb_dimension)
        # word encoder
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension*3)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension*3)
        self.T = nn.Parameter(torch.randn(emb_dimension,emb_dimension,emb_dimension*3))
        self.B = nn.Parameter(torch.randn(emb_dimension,emb_dimension*3))

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def word_embedding(self, pos_u,timevec =None):
        emb_u = self.u_embeddings(pos_u)
        trans_w = torch.einsum('ijk,bk->bij', self.T, emb_u)
        h3 = torch.einsum('bij,bi->bj', trans_w, timevec)
        use_w = self.dense4(h3)
        return use_w

    def time_encoding(self,time):
        h1 = torch.tanh(self.dense1(time.unsqueeze(-1).repeat(1, self.emb_dimension).float()))
        timevec = torch.tanh(self.dense2(h1))
        return  timevec
    def forward(self, pos_u, pos_v, neg_v,time=None):
        # encoding target for context
        timevec = self.time_encoding(time)
        use_w = self.word_embedding(pos_u,timevec)

        #encoding target for postive
        emb_v = self.v_embeddings(pos_v)
        trans_w_v = torch.einsum('ijk,bk->bij', self.T, emb_v)
        h3_v = torch.einsum('bij,bi->bj', trans_w_v, timevec)
        use_c_v = self.dense4(h3_v)

        # encoding targets for negative
        emb_v_neg = self.v_embeddings(neg_v)
        trans_w_v_neg = torch.einsum('ijk,blk->blij', self.T, emb_v_neg) # l is the numbers of nagetive samples
        h3_v_neg = torch.einsum('blij,bli->blj', trans_w_v_neg, timevec.unsqueeze(-2).repeat(1,neg_v.size(1) ,1))
        use_c_v_neg = self.dense4(h3_v_neg)

        score = torch.sum(torch.mul(use_w, use_c_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(use_c_v_neg, use_w.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.mean(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score), torch.mean(score), torch.mean(neg_score)

def fake_dataset(batch_size = 8, number_ns =5):
    while True:
        context = torch.LongTensor(batch_size) % 100
        target =  torch.LongTensor(batch_size) % 100
        negative_samples = torch.LongTensor(batch_size,number_ns) % 100
        time = torch.randn(batch_size)
        yield context,target,negative_samples,time

def main():
    model = Model(101,50)        #  vocabuliary size 101 and dimension 50
    for context,target,negative_samples,time in fake_dataset():
        loss,pos,neg = model(context,target,negative_samples,time)
        print(loss)

if __name__ == '__main__':
    main()
