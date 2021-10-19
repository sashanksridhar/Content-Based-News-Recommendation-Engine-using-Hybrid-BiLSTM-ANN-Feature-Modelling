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
model=load_model("../SingleType/L10Hypernym.h5")

print(model.summary())

rowsxn1 = []
rowsxn2 = []

c = 0
with open("../testWordsL10.csv", 'r', encoding='latin1') as csv1:
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



        del (row[0])
    # print(rows1)
        if c==0:
            rowsxn1.append(rows1)
        elif c==3:
            rowsxn2.append(rows1)
        elif c==4:
            break
        c+=1

        # print(len(rows))


c= 0

rowsx1n1 = []
rowsx1n2 = []


with open("../testHypernymsL10.csv", 'r', encoding='latin1') as csv1:
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

        if c==0:
            rowsx1n1.append(rows1)
        elif c==3:
            rowsx1n2.append(rows1)
        elif c == 4:
            break
        c+=1

    rowsx = []

    with open("../trainWordsL10.csv", 'r', encoding='latin1') as csv1:
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

            del (row[0])
            # print(rows1)

            rowsx.append(rows1)

            # print(len(rows))

    rowsx1 = []

    with open("../trainHypernymsL10.csv", 'r', encoding='latin1') as csv1:
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

    t = Tokenizer()

    t.fit_on_texts(rowsx)
    vocab_size = len(t.word_index) + 1


    embeddings = Word2Vec(size=200, min_count=3)
    embeddings.build_vocab([sentence for sentence in rowsx1])
    embeddings.train([sentence for sentence in rowsx1],
                     total_examples=embeddings.corpus_count,
                     epochs=embeddings.epochs)
    # print(embeddings.wv.most_similar('economy'))

    gen_tfidf = TfidfVectorizer(analyzer=lambda x: x, min_df=3)
    matrix = gen_tfidf.fit_transform([sentence for sentence in rowsx1])
    tfidf_map = dict(zip(gen_tfidf.get_feature_names(), gen_tfidf.idf_))



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


encoded_train_set = t.texts_to_sequences(rowsxn1)
SEQ_LEN = 80

padded_train = pad_sequences(encoded_train_set, maxlen=SEQ_LEN, padding='post')

train_docs = [list(doc) for doc in padded_train]
x_train = np.array([np.array(token) for token in train_docs])
x_train1 = scale(np.concatenate([encode_sentence(ele, 200) for ele in map(lambda x: x, rowsx1n1)]))

encoded_train_set = t.texts_to_sequences(rowsxn2)
SEQ_LEN = 80

padded_train = pad_sequences(encoded_train_set, maxlen=SEQ_LEN, padding='post')

train_docs = [list(doc) for doc in padded_train]
x_trainn2 = np.array([np.array(token) for token in train_docs])
x_train1n2 = scale(np.concatenate([encode_sentence(ele, 200) for ele in map(lambda x: x, rowsx1n2)]))

preds = model.predict([x_train,x_train1])
pred1 = model.predict([x_trainn2,x_train1n2])


i = np.argmax(preds)

cam = GradCAMS(model, i)
i = np.argmax(pred1)
cam_t = GradCAMS(model, i)
val1 = cam.compute_heatmap([x_train,x_train1])
val_t = cam_t.compute_heatmap([x_trainn2,x_train1n2])


a_sparse, b_sparse = sparse.csr_matrix(val1), sparse.csr_matrix(val_t)



# with open ("sample.csv", 'w', encoding='latin1') as csvWrite:
#     filewriter = csv.writer(csvWrite, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
#     l = []
#     for i in range(80):
#         l.append(a_sparse[0,i])
#     filewriter.writerow(l)
sim_sparse = cosine_similarity(a_sparse, b_sparse, dense_output=False)
print(sim_sparse[0,0])
l1 = []
l2 = []












