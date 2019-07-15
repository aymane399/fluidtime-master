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
        self.data18 = gensim.utils.simple_preprocess(open("dta18.txt",'r',errors='ignore').read())
        self.data19 = gensim.utils.simple_preprocess(open("dta19.txt",'r',errors='ignore').read())
        self.word_index = dict([(y,x+1) for x,y in enumerate(sorted(set(self.data18+self.data19)))])
        self.step = 0
        self.target_words = [self.word_index[x] for x in ['vorwort','donnerwetter','presse','feine','anstalt','feder',
        'billig','motiv','anstellung','packen','locker','technisch','geharnischt','zufall','bilanz','englisch','reichstag','museum','abend']]
        super().__init__(batch_size)
        self.vocab_size = vocab_size

    def get_batch(self):
        sequence_int_18 = [self.word_index[x] for x in self.data18[self.step:self.step +250]]
        sequence_int_19 = [self.word_index[x] for x in self.data19[self.step:self.step+250]]
        data18 = skipgrams(sequence_int_18,self.vocab_size)
        data19 = skipgrams(sequence_int_19,self.vocab_size)
        target_indices = [data18[0][i][0] for i in range(len(data18[0]))] +[data19[0][i][0] for i in range(len(data19[0]))]
        context_indices = [data18[0][i][1] for i in range(len(data18[0]))] + [data19[0][i][1] for i in range(len(data19[0]))]
        times = [0]*(len(data18[0])) + [1]*(len(data18[0]))
        labels = data18[1]+data19[1]
        self.step += 250
        return target_indices, context_indices, times, labels
