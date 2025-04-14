import regex
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import math
import pickle

def preprocessQuery(query):
    stopWords = []
    with open("Stopword-List.txt","r") as f:
        data = f.read()
    words = data.split('\n')
    for word in words:
        cleaned = word.strip()
        if cleaned:
            stopWords.append(cleaned)

    query = query.lower()
    query = regex.sub('[^A-Za-z]+',' ',query)
    tokens = word_tokenize(query)
    tokens = [token for token in tokens if token.lower() not in stopWords]

    lemmatizer = WordNetLemmatizer()
    lemmatizedWords = []
    for token in tokens:
        lemmatizedWords.append(lemmatizer.lemmatize(token))

    return lemmatizedWords

def ComputeQueryVector(query,term_list,idf):
    querytf = {}
    for term in query:
        if term not in querytf:
            querytf[term] = 1
        else:
            querytf[term] += 1
    
    queryVector = [0]*len(term_list)
    for term,freq in querytf.items():
        if term in idf:
            termIndex = term_list.index(term)
            normalizedTf = 1+math.log10(freq)
            queryVector[termIndex] = normalizedTf*idf[term]

    return queryVector

with open("tfidf_data.pkl","rb") as f:
    tfidf_data = pickle.load(f)
    termList = tfidf_data['term_list']
    idf = tfidf_data['idf']

query = 'weak heuristic'
processed = preprocessQuery(query)
queryVector = ComputeQueryVector(processed,termList,idf)
print(queryVector)



    

