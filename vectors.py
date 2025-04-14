#Building Vocabulary
import numpy as np
import os
import nltk
from nltk.tokenize import word_tokenize
import json


tf = {} #{doc_id:{term:freq}}

files = os.listdir("Abstracts")
filePairs = [] #(1,'1.txt')

for file in files:
    num = int(file.replace('.txt',''))
    filePairs.append((num,file))

filePairs.sort()

sortedFiles = []
for i in filePairs:
    sortedFiles.append(i[1])

for file in sortedFiles:
    docID = int(file.replace('.txt',''))
    with open("Abstracts/"+file) as f:
        data = f.read()
    tokens = word_tokenize(data)
    if docID not in tf:
        tf[docID] = {}
    for token in tokens:
        if token not in tf[docID]:
            tf[docID][token] = 1
        else:
            tf[docID][token] += 1

print("--------------------TF Computed---------------------------")

df = {} #{term:doc_count}

for file in sortedFiles:
    docID = int(file.replace('.txt',''))
    with open("Abstracts/"+file) as f:
        data = f.read()
    tokens = word_tokenize(data)
    uniqueTokens = set(tokens)
    for token in uniqueTokens:
        if token not in df:
            df[token] = 1
        else:
            df[token] += 1


print("--------------------DF Computed---------------------------")

idf = {} #{term: math.log(N/df[term])}

N = len(sortedFiles)

for term in df:
    idf[term] = np.log10(N/df[term])


print("--------------------IDF Computed---------------------------")

tfidf = {} #{doc_id:{term:tf[doc_id][term]*idf[term]}}

for docID in tf:
    tfidf[docID] = {}
    for term in tf[docID]:
        if term in idf:
            tfidf[docID][term] = tf[docID][term] * idf[term]

print("--------------------TFIDF Computed---------------------------")

#Saving TF IDF Index to a text file

with open("22K4036_tfidf.txt","w") as f:
    json.dump(tfidf,f)

print("--------------------TFIDF Index Saved---------------------------")

