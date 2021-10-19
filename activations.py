# import the necessary packages
from similarity.GradCAM import GradCAMS
from os import walk
import os
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import pandas as pd
import shutil
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import numpy as np
import time
from sklearn.preprocessing import scale
import csv
# construct the argument parser and parse the arguments

from tensorflow.keras.models import load_model
model=load_model("../SingleType/L123Hypernyms.h5")
df = pd.read_csv("../news_train.csv", encoding='latin1')

headlines = df["title"].values.tolist()
# classes = df["class"].values.tolist()
rowsx = []
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []
y6 = []


with open("../trainWords.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(1, len(row)):

            for j in row[i].split("\n"):
                # s+=j+" "

                    # print(j)
                    rows1.append(j)

        if row[0] == "cinema":
            y1.append(1)
            y2.append(0)
            y3.append(0)
            y4.append(0)
            y5.append(0)
            y6.append(0)

        elif row[0] == "lifestyle":
            y1.append(0)
            y2.append(1)
            y3.append(0)
            y4.append(0)
            y5.append(0)
            y6.append(0)

        elif row[0] == "crime":
            y1.append(0)
            y2.append(0)
            y3.append(1)
            y4.append(0)
            y5.append(0)
            y6.append(0)

        elif row[0] == "politics":
            y1.append(0)
            y2.append(0)
            y3.append(0)
            y4.append(1)
            y5.append(0)
            y6.append(0)

        elif row[0] == "science":
            y1.append(0)
            y2.append(0)
            y3.append(0)
            y4.append(0)
            y5.append(1)
            y6.append(0)

        elif row[0] == "business":
            y1.append(0)
            y2.append(0)
            y3.append(0)
            y4.append(0)
            y5.append(0)
            y6.append(1)


        del (row[0])
    # print(rows1)

        rowsx.append(rows1)

        # print(len(rows))


rowsx1 = []


with open("../trainHyponyms.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(0, len(row)):

            for j in row[i].split("\n"):

                    rows1.append(j)



        # del (row[0])
    # print(rows1)

        rowsx1.append(rows1)


rowsx1L2 = []


with open("../trainHyponymsL2.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(0, len(row)):

            for j in row[i].split("\n"):

                    rows1.append(j)



        # del (row[0])
    # print(rows1)

        rowsx1L2.append(rows1)


rowsx1L3 = []


with open("../trainHyponymsL3.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)

    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        s = ''
        # print(row)
        for i in range(0, len(row)):

            for j in row[i].split("\n"):

                    rows1.append(j)



        # del (row[0])
    # print(rows1)

        rowsx1L3.append(rows1)



t = Tokenizer()

t.fit_on_texts(rowsx)
vocab_size = len(t.word_index) + 1
print(vocab_size)
encoded_train_set = t.texts_to_sequences(rowsx)
SEQ_LEN = 80


padded_train = pad_sequences(encoded_train_set, maxlen=SEQ_LEN, padding='post')

train_docs = [list(doc) for doc in padded_train]

embeddings = Word2Vec(size=200, min_count=3)
embeddings.build_vocab([sentence for sentence in rowsx1])
embeddings.train([sentence for sentence in rowsx1],
                 total_examples=embeddings.corpus_count,
                 epochs=embeddings.epochs)
# print(embeddings.wv.most_similar('economy'))

gen_tfidf = TfidfVectorizer(analyzer=lambda x: x, min_df=3)
matrix = gen_tfidf.fit_transform([sentence   for sentence in rowsx1])
tfidf_map = dict(zip(gen_tfidf.get_feature_names(), gen_tfidf.idf_))
print(len(tfidf_map))



embeddingsL2 = Word2Vec(size=200, min_count=3)
embeddingsL2.build_vocab([sentence for sentence in rowsx1L2])
embeddingsL2.train([sentence for sentence in rowsx1L2],
                 total_examples=embeddingsL2.corpus_count,
                 epochs=embeddingsL2.epochs)
# print(embeddings.wv.most_similar('economy'))

gen_tfidfL2 = TfidfVectorizer(analyzer=lambda x: x, min_df=3)
matrixL2 = gen_tfidfL2.fit_transform([sentence   for sentence in rowsx1L2])
tfidf_mapL2 = dict(zip(gen_tfidfL2.get_feature_names(), gen_tfidfL2.idf_))
print(len(tfidf_mapL2))



embeddingsL3 = Word2Vec(size=200, min_count=3)
embeddingsL3.build_vocab([sentence for sentence in rowsx1L3])
embeddingsL3.train([sentence for sentence in rowsx1L3],
                 total_examples=embeddingsL3.corpus_count,
                 epochs=embeddingsL3.epochs)
# print(embeddings.wv.most_similar('economy'))

gen_tfidfL3 = TfidfVectorizer(analyzer=lambda x: x, min_df=3)
matrixL3 = gen_tfidfL3.fit_transform([sentence   for sentence in rowsx1L3])
tfidf_mapL3 = dict(zip(gen_tfidfL3.get_feature_names(), gen_tfidfL3.idf_))
print(len(tfidf_mapL3))


def encode_sentence(tokens, emb_size):
    _vector = np.zeros((1, emb_size))
    length = 0
    for word in tokens:
        try:
            _vector += embeddings.wv[word].reshape((1, emb_size)) * tfidf_map[word]
            length += 1
        except KeyError:
            continue
        break

    if length > 0:
        _vector /= length

    return _vector


def encode_sentenceL2(tokens, emb_size):
    _vector = np.zeros((1, emb_size))
    length = 0
    for word in tokens:
        try:
            _vector += embeddingsL2.wv[word].reshape((1, emb_size)) * tfidf_mapL2[word]
            length += 1
        except KeyError:
            continue
        break

    if length > 0:
        _vector /= length

    return _vector



def encode_sentenceL3(tokens, emb_size):
    _vector = np.zeros((1, emb_size))
    length = 0
    for word in tokens:
        try:
            _vector += embeddingsL3.wv[word].reshape((1, emb_size)) * tfidf_mapL3[word]
            length += 1
        except KeyError:
            continue
        break

    if length > 0:
        _vector /= length

    return _vector


with open ("activationsNew.csv", 'w', encoding='latin1') as csvWrite:
    filewriter = csv.writer(csvWrite, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')


    for i in range(0,len(rowsx1)):
        l0 = []
        l0.append(rowsx[i])
        print(i)
        l1 = []
        l1.append(rowsx1[i])
        l2 = []
        l2.append(rowsx1L2[i])
        l3 = []
        l3.append(rowsx1L3[i])

        encoded_train_set = t.texts_to_sequences(l0)
        SEQ_LEN = 80

        padded_train = pad_sequences(encoded_train_set, maxlen=SEQ_LEN, padding='post')

        train_docs = [list(doc) for doc in padded_train]
        x_train = np.array([np.array(token) for token in train_docs])
        x_train1 = scale(np.concatenate([encode_sentence(ele, 200) for ele in map(lambda x: x, l1)]))


        x_train3 = scale(np.concatenate([encode_sentenceL2(ele, 200) for ele in map(lambda x: x, l2)]))


        x_train5 = scale(np.concatenate([encode_sentenceL3(ele, 200) for ele in map(lambda x: x, l3)]))


        preds = model.predict([x_train,x_train1,x_train3,x_train5])


        ind = np.argmax(preds)
        cam = GradCAMS(model, ind)

        val1 = cam.compute_heatmap([x_train,x_train1,x_train3,x_train5])



        a_sparse = sparse.csr_matrix(val1)

        a1 =[]
        a1.append(headlines[i])

        for i in range(50):
            a1.append(a_sparse[0, i])
        filewriter.writerow(a1)









