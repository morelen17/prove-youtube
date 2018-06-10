from __future__ import print_function

import collections
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from six.moves import cPickle as pickle


class Paragraph2Vec:
    """
    Shape of word matrix W is (self._vocabulary_size x self._embedding_size)
    Shape of paragraph matrix D is (self._paragraph_size x self._embedding_size)

    """

    def __init__(self, is_training: bool = True):
        self._is_training = is_training
        self._embedding_size = 50
        self._window_size = 8
        self._vocabulary_size = 49999
        self._unk_id = 50000  # ID of UNK word token, for padding

        self._id_to_paragraph = None  # dict: {0: ['youtube', 'video', ...], ...}
        self._word_to_id = None  # {'youtube': 0, 'video': 1, ..}

        self._paragraphs_num = 0
        self._total_words_num = 0  # sum(len(paragraph)) for each paragraph

        self._neg_samples_num = 64
        self._batch_size = 128
        self._learning_rate = 0.001
        self._num_epoch = 50

        self.x, self.y = None, None
        self.loss_op, self.train_op = None, None
        self.paragraph_matrix, self.word_matrix = None, None
        pass

    def build_vocab(self, paragraph_list: list):
        paragraph_list = list(map(lambda x: x.lower().split(' '), paragraph_list))
        vocabulary = collections.Counter()
        self._paragraphs_num = len(paragraph_list)

        self._id_to_paragraph = dict(zip(list(range(self._paragraphs_num)), paragraph_list))

        stop_words = set(stopwords.words('english'))
        for paragraph in paragraph_list:
            self._total_words_num += len(paragraph)
            for word in paragraph:
                if word != '' and len(word) > 1 and word not in stop_words:
                    vocabulary[word] += 1

        self._word_to_id = {val[0]: key for key, val in enumerate(vocabulary.most_common(self._vocabulary_size))}
        del vocabulary, stop_words, paragraph_list
        return self

    def build_model(self):
        self.x = tf.placeholder(tf.int32, shape=[self._batch_size, self._window_size + 1])
        self.y = tf.placeholder(tf.int32, shape=[self._batch_size, 1])

        self.paragraph_matrix = tf.Variable(tf.random_uniform([self._paragraphs_num, self._embedding_size], -1.0, 1.0))
        self.word_matrix = tf.Variable(tf.random_uniform([self._vocabulary_size, self._embedding_size], -1.0, 1.0))

        concatenated = []
        for i in range(self._window_size):
            concatenated.append(tf.nn.embedding_lookup(self.word_matrix, self.x[:, i]))
        concatenated.append(tf.nn.embedding_lookup(self.paragraph_matrix, self.x[:, self._window_size]))

        concatenated_matrix = tf.concat(concatenated, 1)
        concatenated_matrix_length = (1 + self._window_size) * self._embedding_size
        u_matrix = tf.Variable(tf.truncated_normal([self._vocabulary_size, concatenated_matrix_length],
                                                   stddev=1.0 / np.sqrt(concatenated_matrix_length)))
        b_bias = tf.Variable(tf.zeros([self._vocabulary_size]))

        if self._is_training:
            self.loss_op = tf.nn.sampled_softmax_loss(weights=u_matrix,
                                                      biases=b_bias,
                                                      labels=self.y,
                                                      inputs=concatenated_matrix,
                                                      num_sampled=self._neg_samples_num,
                                                      num_classes=self._vocabulary_size)
        else:
            self.loss_op = tf.nn.nce_loss(weights=u_matrix,
                                          biases=b_bias,
                                          labels=self.y,
                                          inputs=concatenated_matrix,
                                          num_sampled=self._neg_samples_num,
                                          num_classes=self._vocabulary_size)

        self.loss_op = tf.reduce_mean(self.loss_op)
        self.train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(self.loss_op)
        return self

    def _paragraph_generator(self):
        """
        Yields paragraph id and list of word ids of this paragraph.
        """

        def paragraph_to_word_ids(paragraph: list):
            return [self._word_to_id[i] for i in paragraph if i in self._word_to_id]

        for par_id in random.sample(range(self._paragraphs_num), self._paragraphs_num):
            yield par_id, paragraph_to_word_ids(self._id_to_paragraph[par_id])

    def _context_generator(self, paragraph: list):
        """
        Yields center word id and its context.
        :param paragraph: list of word ids
        """
        half_window = self._window_size // 2
        paragraph.extend(half_window * [self._unk_id])
        paragraph = (half_window * [self._unk_id]) + paragraph
        for i in range(len(paragraph) - self._window_size):
            yield paragraph[half_window + i], \
                  paragraph[i:half_window + i] + paragraph[half_window + i + 1:half_window * 2 + i + 1]

    def _sample_generator(self):
        """
        Yields target word id and list of context word ids plus paragraph id.
        """
        for par_id, paragraph in self._paragraph_generator():
            for target_word_id, context_list in self._context_generator(paragraph):
                yield target_word_id, context_list + [par_id]

    def _batch_generator(self):
        current_step = 0
        x_batch, y_batch = [], []
        for target_word_id, context_and_par_id in self._sample_generator():
            current_step += 1
            y_batch.append(target_word_id)
            x_batch.append(context_and_par_id)
            if current_step == self._batch_size:
                yield x_batch, y_batch
                x_batch, y_batch = [], []
                current_step = 0

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            global_step = 1
            global_loss = 0
            for epoch in range(1, self._num_epoch + 1):
                print("Epoch:", str(epoch))
                for x_batch, y_batch in self._batch_generator():
                    loss, _ = sess.run(
                        [self.loss_op, self.train_op],
                        feed_dict={
                            self.x: x_batch,
                            self.y: np.array(y_batch).reshape(self._batch_size, 1)
                        })
                    global_loss += loss
                    if global_step % 1000 == 0:
                        print("Step", str(global_step),
                              "| Batch loss = {:.4f}".format(loss),
                              "| Global loss = {:.4f}".format(global_loss / global_step))
                    global_step += 1
                print("End of epoch %d (step %d)\n-------" % (epoch, global_step))
            return self.paragraph_matrix.eval(sess), self.word_matrix.eval(sess)


if __name__ == '__main__':
    start = time.time()
    df = pd.read_csv('./data_preprocessed/Allvideos.csv')
    print('Data reading complete, time (s):', time.time() - start)
    start = time.time()
    model = Paragraph2Vec().build_vocab(df['text'].tolist()).build_model()
    print('Model build completed, time (s):', time.time() - start)
    start = time.time()
    paragraph_matrix, word_matrix = model.train()
    print('Train completed, time (s):', time.time() - start)
    print(paragraph_matrix.shape, word_matrix.shape)
    with open('./data_preprocessed/paragraph_matrix.pkl', 'wb') as f:
        pickle.dump(paragraph_matrix, f)
    with open('./data_preprocessed/word_matrix.pkl', 'wb') as f:
        pickle.dump(word_matrix, f)
    print('Fin!')
