import random
from tqdm import tqdm
import gensim
from keras.preprocessing.sequence import skipgrams
import numpy as np

from iterators.DataIterator import DataTterator

class DumbIterator(DataTterator):
    """This class is to demonstrate the format that your DataIterator should produce.
    """
    

    def __init__(self, batch_size, vocab_size):
        self.batch_size = batch_size
        self.data18 = gensim.utils.simple_preprocess(open("dta18.txt",'r',errors='ignore').read())
        self.data19 = gensim.utils.simple_preprocess(open("dta19.txt",'r',errors='ignore').read())
        self.predata18 = open("dta18.txt",'r',errors='ignore')
        self.predata19 = open("dta19.txt",'r',errors='ignore')
        self.target_indices = []
        self.context_indices = []
        self.times = []
        self.labels = []
        self.word_index = dict([(y,x+1) for x,y in enumerate(sorted(set(self.data18+self.data19)))])
        self.target_words = [self.word_index[x] for x in ['vorwort','donnerwetter','presse','feine','anstalt','feder',
        'billig','motiv','anstellung','packen','locker','technisch','geharnischt','zufall','bilanz','englisch','reichstag','museum','abend']]
        super().__init__(batch_size)
        self.vocab_size = vocab_size

    def get_batch(self):
        s = 0
        while (len(self.target_indices) < self.batch_size) and s==0 :
            s = 0
            line18 = self.predata18.readline()
            if line18 == '':
                s += 1
            else :
                time18 = float((int(line18[0:4])-1750)/(1799-1750))
                sequence_int_18 = [self.word_index[x] for x in gensim.utils.simple_preprocess(line18)]
                data18 = skipgrams(sequence_int_18,self.vocab_size,window_size=2)
            line19 = self.predata18.readline()            
            if line19 =='':
                s += 1
            else: 
                time19 = float((int(line19[0:4])-1850)/(1899-1850))            
                sequence_int_19 = [self.word_index[x] for x in gensim.utils.simple_preprocess(line19)]
                data19 = skipgrams(sequence_int_19,self.vocab_size,window_size=2)
            if s == 0:

                self.target_indices += [data18[0][i][0] for i in range(len(data18[0]))] +[data19[0][i][0] for i in range(len(data19[0]))]
                self.context_indices += [data18[0][i][1] for i in range(len(data18[0]))] + [data19[0][i][1] for i in range(len(data19[0]))]
                self.times += [time18]*(len(data18[0])) + [time19]*(len(data19[0]))
                self.labels += data18[1]+data19[1]

        target_indices = self.target_indices[0:self.batch_size]
        self.target_indices = self.target_indices[self.batch_size:]
        context_indices = self.context_indices[0:self.batch_size]
        self.context_indices = self.context_indices[self.batch_size:]
        times = self.times[0:self.batch_size]
        self.times = self.times[self.batch_size:]
        labels = self.labels[0:self.batch_size]
        self.labels = self.labels[self.batch_size:]

        return target_indices, context_indices, times, labels
