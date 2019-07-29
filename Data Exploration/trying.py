#http://adventuresinmachinelearning.com/gensim-word2vec-tutorial/
#https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

new_sentence = ['DHCP', 'ACK', 'from', 'AP', 'has', 'wrong', 'DS', 'address', 'in', 'capwap', 'header']
a= "DHCP ack sent to WLC AP was connected to previously"
words = list(model.wv.vocab)
matrix1=[]
for word in new_sentence:
    if(word in words):
        matrix1.append(model[word.lower()])

import scipy.spatial as sp
results2 = 1 - sp.distance.cdist(matrix1, matrix2, 'cosine')
