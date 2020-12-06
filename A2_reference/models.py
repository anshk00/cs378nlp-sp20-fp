# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class DANN(nn.Module):
    def __init__(self, inp, hid, out, word_embeddings):
        super(DANN, self).__init__()
        for vec in word_embeddings.vectors:
            for i in range(len(vec)):
                vec[i]=random.uniform(-1,1)

        self.preEmbeds = nn.Embedding.from_pretrained(torch.Tensor(word_embeddings.vectors).float())
        self.V = nn.Linear(inp, hid)
        self.g = nn.Tanh()
        # self.g = nn.ReLU()
        self.W = nn.Linear(hid, out)
        self.log_softmax = nn.LogSoftmax(dim=1)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, wordindices:List[List[int]]) -> List[int]:
        torc = torch.empty(len(wordindices),300)
        # print(torc.size())
        for j in range(len(wordindices)):
            sum = torch.tensor([np.array(self.preEmbeds(i)) for i in wordindices[j]])
            # print(len(wordindices[0]))
            # print(sum.size())
            torcy = torch.sum(sum,0)
            torc[j] = torcy
        # print(torc[j])
        res = self.log_softmax(self.W(self.g(self.V(torc))))
        # print(res)
        # raise NotImplementedError
        # print(res.size())
        return res

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, dann, word_embeddings:WordEmbeddings):
        self.dann = dann
        self.word_embeddings = word_embeddings
        self.log_softmax = nn.LogSoftmax(dim=0)
    def predict(self, ex_words: List[str]) -> int:
        sum = torch.Tensor(np.array([self.word_embeddings.get_embedding(word) for word in ex_words]))
        torc = torch.sum(sum,0)
        softmax = torch.neg(self.log_softmax(self.dann.W(self.dann.g(self.dann.V(torc)))))   
        if (softmax[0] > softmax[1]):
            return 1
        else:
            return 0


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    random.seed()
    input_size = 50 if (args.word_vecs_path=='data/glove.6B.50d-relativized.txt') else 300
    hidden_size = args.hidden_size
    output_size = 2
    batch_size = args.batch_size
    dann = DANN(input_size,hidden_size,output_size,word_embeddings)
    num_epochs = args.num_epochs
    learning_rate = args.lr
    nnfun = nn.NLLLoss(reduction='sum')
    optimizer = optim.Adam(dann.parameters(),lr=learning_rate)
    for gen in range(num_epochs):
        ex_indices = [i for i in range(0, len(train_exs))]
        random.shuffle(ex_indices) #"batch" them here
        total_loss = 0.0
        for idx in range(int(len(ex_indices)/batch_size)): #iterate through batches
            if((idx+1)*batch_size<len(ex_indices)):
                x = [[]for piece in range(batch_size)]
                y = torch.tensor(np.zeros(batch_size)).long()
            else:
                x = [[]for piece in range(len(ex_indices)-(idx*batch_size))]
                y = torch.tensor(np.zeros(len(ex_indices)-(idx*batch_size))).long()
            # print(x)
            for piece in range(len(x)):
                for word in train_exs[ex_indices[idx*batch_size+piece]].words: #do this for each sentence in batch
                    ind = word_embeddings.word_indexer.index_of(word)
                    if ind != -1:
                        x[piece].append(ind)
                    else:
                        x[piece].append(1)
                # print(x)
                x[piece] = torch.tensor(x[piece]).long()
                # print(x)
                y[piece] = train_exs[ex_indices[idx*batch_size+piece]].label #expand to branch_size length array
            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
            dann.zero_grad()
            log_probs = dann.forward(x)
            # print(y)
            # print('log probs: ',log_probs)
            # print('y: ', y)
            loss = nnfun(log_probs,y)
            total_loss += loss
            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (gen, total_loss))
    return NeuralSentimentClassifier(dann,word_embeddings)

