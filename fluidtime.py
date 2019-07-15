import numpy as np
import tensorflow as tf
from ModelEval import ModelEval
from ModelTrainer import ModelTrainer
from iterators.DumbIterator import DumbIterator
from models.DiffTime import DiffTime

vocab_size = 40752
batch_size = 7920
# num_iterations = 99000
num_iterations = 92671
num_iter_per_epoch = 100




def cos(A,B):
    A = A.reshape(-1)
    B = B.reshape(-1)
    a = 0 
    b = 0
    c = 0
    for i in range(len(A)):
        a += A[i]**2
        b += B[i]**2
        c += A[i]*B[i]
    a = a**0.5
    b = b**0.5
    return (c)/(a*b)

dataiter = DumbIterator(batch_size, vocab_size)

with tf.Session() as sess:
    model = DiffTime(vocab_size)
    trainer = ModelTrainer(sess, dataiter, model)
    trainer.train_model(batch_size, num_iterations, num_iter_per_epoch)
    modeleval = ModelEval(sess, model)
    print(type(model.get_target_vector([1],[0.0]).eval()))
    L = [cos(model.get_target_vector([x],[0.0]).eval(),model.get_target_vector([x],[1.0]).eval()) for x in trainer.dataiter.target_words] 

print(L)
np.save('drift',np.array(L))