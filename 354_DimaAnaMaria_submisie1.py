import random as random
import re
from collections import Counter
import nltk
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from datetime import datetime
from sklearn.metrics import f1_score, confusion_matrix
nltk.download('stopwords')


def tokenize(text):
    text = text.lower()  # face toate literele mici
    text = re.sub('[^A-Za-z]', ' ', text)  # scoate caracterele nonalfabetice
    """stops = set(stopwords.words("italian"))
    text = gensim.corpora.textcorpus.strip_multiple_whitespaces(text)
    filtered_words = [word for word in text.split() if word not in stops]
    filtered_words = gensim.corpora.textcorpus.remove_short(filtered_words, minsize=3)
    text = " ".join(filtered_words)
    text = gensim.parsing.preprocessing.strip_punctuation2(text)
    text = gensim.parsing.preprocessing.strip_numeric(text)
    text = gensim.corpora.textcorpus.strip_multiple_whitespaces(text)
"""
    return nltk.word_tokenize(text)
# reduce len = reduce acelasi caracter la maxim 3(aaaaaa->aaa)
# strip_handles= separa litera si semnul de punctuatie in 2 caractere diferite


def get_vocabulary_from_corpus(corpus):
    # returneaza toate cuvintele din corpus-ul trimis ca parametru
    counter = Counter()
    for i in corpus:
        tokens = tokenize(i)
        counter.update(tokens)
    return counter


def get_representation(vocabulary, how_many):
    '''Extract the first most common words from a vocabulary
    and return two dictionaries: word to index and index to word
    wrd: @  che  .   ,   di  e
    idx: 0   1   2   3   4   5
    '''
    most_comm = vocabulary.most_common(how_many)
    # print(most_comm)
    wd2idx = {}
    idx2wd = {}
    for i, iterator in enumerate(most_comm):
        idx2wd[i] = iterator[0]
        wd2idx[iterator[0]] = i
    # print(most_comm)
    return wd2idx, idx2wd


def text_to_bow(text, wd2idx):
    '''Convert a text to a bag of words representation.
           @  che  .   ,   di  e
    text   0   1   0   2   0   1
    '''
    proprietati = np.zeros(len(wd2idx))
    for token in tokenize(text):

        if token in wd2idx:
            pozitie = wd2idx[token]
            proprietati[pozitie] += 1

    return proprietati


def corpus_to_bow(corpus, wd2idx):
    '''Convert a corpus to a bag of words representation.
           @  che  .   ,   di  e
    text0  0   1   0   2   0   1
    text1  1   2 ...
    ...
    textN  0   0   1   1   0   2
    '''
    aux = []
    for text in corpus:
        aux.append(text_to_bow(text, wd2idx))
    aux = np.array(aux)

    return aux


def split(data, labels, procentaj_valid):
    '''
    important! mai intai facem shuffle la date
    '''
    indici = np.arange(0, len(labels))
    random.shuffle(indici)
    N = int((1 - procentaj_valid) * len(labels))
    train = data[indici[:N]]  # propozitia
    valid = data[indici[N:]]  # testarea
    y_train = labels[indici[:N]]  #  daca e misogina/nemisogina
    y_valid = labels[indici[N:]]  # stim etichetele

    return train, valid, y_train, y_valid


def cross_validate(k, data, labels):
    chunk_size = len(labels) // k
    indici = np.arange(0, len(labels))
    random.shuffle(indici)
    for i in range(0, len(labels), chunk_size):
        valid_indici = indici[i:i + chunk_size]
        train_indici = np.concatenate([indici[0:i], indici[i + chunk_size:]])
        train = data[train_indici]
        valid = data[valid_indici]
        y_train = labels[train_indici]
        y_valid = labels[valid_indici]
        yield train, valid, y_train, y_valid


def scriere(predictions, filename="predictions.csv"):
    import os
    if os.path.exists(filename):
        os.remove(filename)
    f = open(filename, "w")
    f.write("id,label\n")

    for prediction in predictions:
        f.write(str(prediction[0]) + "," + str(prediction[1]) + "\n")
    f.close()


# main


# --------------------------citire date
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
corpus = train_df['text']
labels = train_df['label']
# ------------------------------

# scoate toate cuvintele din corpus
toate_cuvintele = get_vocabulary_from_corpus(corpus)

# scoate reprezentarea pt primele 100 de cuvinte obtinute in urma get_vocabulary_from_corpus si
# returneaza 2 dictionare covant->index si index->cuvant
wd2idx, idx2wd = get_representation(toate_cuvintele, 100)

# transforma corpusul intr-un BoW (fiecare cuvant cu nr lui de aparitii)
data = corpus_to_bow(corpus, wd2idx)

test_data = corpus_to_bow(test_df['text'], wd2idx)
print(test_data.shape)

train, valid, y_train, y_valid = split(data, labels, 0.1)

#clasificator folosit
clf = KNeighborsClassifier(n_neighbors=3, n_jobs=4)


# antrenare model
now1 = datetime.now()
clf.fit(train, y_train)
now2 = datetime.now()
print(now2 - now1)

# clasificatorul prezice
rez_predictii = clf.predict(test_data)
#predictions = clf.predict(valid)

# 10 fold cross validation
#cv_scores = cross_val_score(clf, train, y_train, cv=10, n_jobs=4)
#print(cv_scores)
#print(cv_scores.mean())

scoruri = []
matrici_de_confuzie = []
matrici_de_confuzie_normalizata = []


for train, valid, y_train, y_valid in  cross_validate(10, data, labels):
    clf.fit(train, y_train)
    predictions = clf.predict(valid)
    scor = f1_score(y_valid, predictions)
    print(scor)
    #print("\nMatricea de confuzie medie este: \n", np.sum(matrici_de_confuzie, axis=0))
    scoruri.append(scor)

    conf_matrix = confusion_matrix(y_valid, predictions)
    conf_matrix_normalizata = confusion_matrix(y_valid, predictions,normalize='true')

    matrici_de_confuzie.append(conf_matrix)
    matrici_de_confuzie_normalizata.append(conf_matrix_normalizata/10)


print("\nMatricea de confuzie medie este: \n", np.sum(matrici_de_confuzie,axis=0))
print("\nMatricea de confuzie medie normalizata este: \n", np.sum(matrici_de_confuzie_normalizata,axis=0))



predictii_fisier = []
for i, eticheta in enumerate(rez_predictii):
    predictii_fisier.append((i + 5001, eticheta))
scriere(predictii_fisier, "pred21.csv")


