import time
from tqdm import tqdm
import tensorflow as tf

from iterators.DataIterator import DataTterator
from models.Model import Model


class ModelTrainer(object):
    def __init__(self, sess:tf.Session, dataiter:DataTterator, model:Model):
        self.dataiter = dataiter
        self.model = model
        self.sess = sess
        pass

    def train_model(self, batch_size:int, num_iterations:int, num_iter_per_epoch:int):
        targets_placeholder = tf.placeholder(tf.int32, shape=(batch_size,))
        contexts_placeholder = tf.placeholder(tf.int32, shape=(batch_size,))
        times_placeholder = tf.placeholder(tf.float32, shape=(batch_size,))
        labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size,))

        optimizer = tf.train.AdamOptimizer(1e-4)
        loss = self.model.get_loss(targets_placeholder, contexts_placeholder, times_placeholder, labels_placeholder)
        tf.summary.scalar('loss', loss)
        train_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        start_time = time.time()
        for step in tqdm(range(0, num_iterations)):
            targets, contexts, times, labels = self.dataiter.get_batch()
            if len(targets) == batch_size:
                feed_dict = {
                    targets_placeholder: targets,
                    contexts_placeholder: contexts,
                    times_placeholder: times,
                    labels_placeholder: labels,
                }
                _, loss_value = self.sess.run([train_op, loss],
                                         feed_dict=feed_dict)

                duration = time.time() - start_time

                if step % num_iter_per_epoch == 0:
                    print('Step %d: loss = %.4f (%.3f sec)' % (step, loss_value, duration))
            else :
                print('out of data')
                break
