import pandas as pd
import numpy as np
import csv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.preprocessing import scale
from scipy import sparse
from similarity.GradCAM import GradCAMS

df = pd.read_csv("activationsNew.csv",encoding='latin1')

y_train = df["headline"].values.tolist()
df.drop(columns=["headline"],inplace=True)

from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric = 'cosine', algorithm= 'brute')

XK = df
YK = y_train

model_knn.fit(df,y_train)

from tensorflow.keras.models import load_model
model=load_model("../SingleType/L123Hypernyms.h5")
df = pd.read_csv("../news_test.csv", encoding='latin1')

headlines = df["title"].values.tolist()
classesFinal = df["class"].values.tolist()

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


with open("../trainHypernyms.csv", 'r', encoding='latin1') as csv1:
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


with open("../trainHypernymsL2.csv", 'r', encoding='latin1') as csv1:
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


with open("../trainHypernymsL3.csv", 'r', encoding='latin1') as csv1:
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



rowsx_t_1 = []
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []
y6 = []

testclasses = []
dictcount = {"cinema":{"cinema":0,"lifestyle":0,"crime":0,"science":0,"politics":0,"business":0},
             "lifestyle" : {"cinema": 0,"lifestyle":0, "crime": 0, "science": 0, "politics": 0, "business": 0},
             "crime": {"cinema": 0, "lifestyle" : 0, "crime":0, "science": 0, "politics": 0, "business": 0},
             "science": {"cinema": 0, "lifestyle": 0,"crime" : 0,"science":0, "politics": 0, "business": 0},
             "politics": {"cinema": 0, "lifestyle": 0, "crime": 0,"science" : 0, "politics":0, "business": 0},
             "business": {"cinema": 0, "lifestyle": 0, "crime": 0, "science": 0, "politics": 0,"business":0}}

dictdist = {"cinema":{"cinema":0,"lifestyle":0,"crime":0,"science":0,"politics":0,"business":0},
             "lifestyle" : {"cinema": 0,"lifestyle":0, "crime": 0, "science": 0, "politics": 0, "business": 0},
             "crime": {"cinema": 0, "lifestyle" : 0, "crime":0, "science": 0, "politics": 0, "business": 0},
             "science": {"cinema": 0, "lifestyle": 0,"crime" : 0,"science":0, "politics": 0, "business": 0},
             "politics": {"cinema": 0, "lifestyle": 0, "crime": 0,"science" : 0, "politics":0, "business": 0},
             "business": {"cinema": 0, "lifestyle": 0, "crime": 0, "science": 0, "politics": 0,"business":0}}
with open("../testWords.csv", 'r', encoding='latin1') as csv1:
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

        rowsx_t_1.append(rows1)

        # print(len(rows))





rowsx_t_2 = []


with open("../testHypernyms.csv", 'r', encoding='latin1') as csv1:
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




    # print(rows1)

        rowsx_t_2.append(rows1)


rowsx_t_2L2 = []


with open("../testHypernymsL2.csv", 'r', encoding='latin1') as csv1:
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




    # print(rows1)

        rowsx_t_2L2.append(rows1)


rowsx_t_2L3 = []


with open("../testHypernymsL3.csv", 'r', encoding='latin1') as csv1:
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




    # print(rows1)

        rowsx_t_2L3.append(rows1)

index = 69

totalPrec = 0
totalRec = 0

for index in range(0,len(rowsx_t_1)):
    print(index)

    reqrows_t_1 = []
    reqrows_t_1.append(rowsx_t_1[index-1])


    encoded_train_set = t.texts_to_sequences(reqrows_t_1)
    SEQ_LEN = 80


    padded_train = pad_sequences(encoded_train_set, maxlen=SEQ_LEN, padding='post')

    train_docs = [list(doc) for doc in padded_train]
    x_train = np.array([np.array(token) for token in train_docs])

    reqrows_t_2 = []
    reqrows_t_2.append(rowsx_t_2[index-1])

    x_train1 = scale(np.concatenate([encode_sentence(ele, 200) for ele in map(lambda x: x, reqrows_t_2)]))
    # x_train2 = scale(np.concatenate([encode_sentence1(ele, 200) for ele in map(lambda x: x, rowsx_t_3)]))
    reqrows_t_3 = []
    reqrows_t_3.append(rowsx_t_2L2[index-1])
    x_train3 = scale(np.concatenate([encode_sentenceL2(ele, 200) for ele in map(lambda x: x, reqrows_t_3)]))

    reqrows_t_4 = []
    reqrows_t_4.append(rowsx_t_2L3[index-1])
    # x_train4 = scale(np.concatenate([encode_sentence1L2(ele, 200) for ele in map(lambda x: x, rowsx_t_3L2)]))
    x_train5 = scale(np.concatenate([encode_sentenceL3(ele, 200) for ele in map(lambda x: x, reqrows_t_4)]))
    # x_train6 = scale(np.concatenate([encode_sentence1L3(ele, 200) for ele in map(lambda x: x, rowsx_t_3L3)]))
    preds = model.predict([x_train,x_train1,x_train3,x_train5])

    print(preds)

    classNames = ["cinema", "lifestyle", "crime", "politics", "science", "business"]

    top3 = sorted(zip(preds, classNames), reverse=True)[:3]

    relevant = [i[1] for i in top3]

    ind = np.argmax(preds)
    cam = GradCAMS(model, ind)

    val1 = cam.compute_heatmap([x_train,x_train1,x_train3,x_train5])



    a_sparse = sparse.csr_matrix(val1)

    a1 =[]


    for i in range(50):
        a1.append(a_sparse[0, i])

    # print(headlines[index])
    # print(classesFinal[index])
    classvar = classesFinal[index]

    df = pd.read_csv("../news_train.csv",encoding="latin1")
    classes = df["class"].values.tolist()

    recommended = []

    distances, indices = model_knn.kneighbors(np.array(a1).reshape(1, -1), n_neighbors = 3)
    recClasses = [classes[indices.flatten()[i]] for i in range(0, len(indices.flatten()))]
    recommended = [i[1] for i in sorted(zip(distances.flatten(), recClasses))[:3]]

    for i in range(0, len(distances.flatten())):
        # print(y_train[indices.flatten()[i]])
        # print(classes[indices.flatten()[i]])
        # print(distances.flatten()[i])
        dictcount[classvar][classes[indices.flatten()[i]]]+=1
        dictdist[classvar][classes[indices.flatten()[i]]] += distances.flatten()[i]

    precision = 0
    recall = 0
    if len(recommended)==0:
        precision = 0
        recall = 0
    else:
        num = 0
        for i in recommended:
            if i in relevant:
                num+=1
        precision = num/len(recommended)
        recall = num/len(relevant)

    print("Scores")
    print(precision)
    print(recall)
    totalPrec+=precision
    totalRec+=recall


    print(dictcount)
    print(dictdist)


print("final")
print(dictcount)
print(dictdist)

print(totalPrec/len(rowsx_t_1))
print(totalRec/len(rowsx_t_1))
from mlxtend.plotting import plot_decision_regions
# import matplotlib.pyplot as plt
# fig, ax1 = plt.subplots()
# pca = PCA(n_components=2).fit(XK.to_numpy())
#
#
# data2D = pca.transform(np.array(a1).reshape(1,-1))
# # ax1.scatter(data2D[:,0],data2D[:,1])
# ax1.scatter(data2D[:,0],data2D[:,1],label="input")
# rows = XK.to_numpy()
# add = 0.0001
# for i in range(0, len(distances.flatten())):
#
#     data2D = pca.transform(rows[indices.flatten()[i]].reshape(1, -1))
#     print(i)
#     print(data2D)
#     colors = plt.cm.rainbow(np.linspace(0, 1, len(distances.flatten())))
#     # ax1.scatter(data2D[:,0]+add,data2D[:,1]+add)
#     ax1.scatter(data2D[:,0]+add,data2D[:,1]+add,label=classes[indices.flatten()[i]])
#     add+=0.00001
# ax1.legend()
# plt.show()