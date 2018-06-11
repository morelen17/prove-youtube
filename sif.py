import numpy as np

from paragraph2vec import EMBEDDING_SIZE
from sklearn.decomposition import PCA
from wordfreq import word_frequency


class SIF:
    """
    A Simple but Tough-to-Beat Baseline for Sentence Embeddings (https://openreview.net/pdf?id=SyK00v5xx)
    """
    def __init__(self, word_matrix: dict, paragraph_list: list):
        self._word_matrix = word_matrix
        self._a = 0.001
        self._embedding_size = EMBEDDING_SIZE
        self._language = 'en'
        self._paragraph_list = paragraph_list
        self._paragraph_matrix = np.zeros((len(paragraph_list), self._embedding_size))

    def _get_paragraph_vector(self, paragraph: list):
        vector = np.zeros(self._embedding_size)
        sentence_len = 0
        for word in paragraph:
            if word in self._word_matrix:
                weight = self._a / (self._a + word_frequency(word, self._language))
                vector = np.add(vector, np.multiply(weight, self._word_matrix[word]))
                sentence_len += 1
        vector = np.divide(vector, sentence_len)
        return vector

    def get_paragraph_matrix(self):
        for key, par_list in enumerate(self._paragraph_list):
            self._paragraph_matrix[key] = self._get_paragraph_vector(self._paragraph_list[key])

        pca = PCA(n_components=self._embedding_size)
        pca.fit(self._paragraph_matrix)
        u_vector = pca.components_[0]

        self._paragraph_matrix = np.apply_along_axis(lambda vector: vector - u_vector * u_vector.transpose() * vector,
                                                     axis=1,
                                                     arr=self._paragraph_matrix)
        return self._paragraph_matrix
