import pickle
import math
from queryProcessing import preprocessQuery,ComputeQueryVector

def loadTfidfData(tfidfPath='tfidf_data.pkl'):
    with open(tfidfPath,"rb") as f:
        data = pickle.load(f)
    return data['term_list'],data['idf'],data['tfidf_vectors']

def cosineSimilarity(queryVector,tfidfVectors):
    dotProduct = sum(a*b for a,b in zip(queryVector,tfidfVectors))
    normA = math.sqrt(sum(a*a for a in queryVector))
    normB = math.sqrt(sum(b*b for b in tfidfVectors))

    return dotProduct/(normA*normB) if (normA*normB) != 0 else 0

def rankDocs(query,term_list,idf,tfidf_vectors,alpha):
    processedQuery = preprocessQuery(query)
    queryVector = ComputeQueryVector(processedQuery,term_list,idf)
    results = []
    for doc_id,docVector in tfidf_vectors.items():
        similarity = cosineSimilarity(queryVector,docVector)
        if similarity>=alpha:
            results.append((doc_id,similarity))
    results.sort(key=lambda x: x[1],reverse=True)
    return results if len(results)<100 else results[:100]

term_list,idf,tfidf_vectors = loadTfidfData()
query = 'weak heuristic'
results = rankDocs(query,term_list,idf,tfidf_vectors,0.001)
print("Ranked Results for Query: ",query)

for rank,(doc_id,similarity) in enumerate(results,1):
     print(f"{rank}. Document {doc_id} (score: {similarity:.4f})")