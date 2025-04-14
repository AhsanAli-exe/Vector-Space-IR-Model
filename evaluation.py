from computeSimilarity import loadTfidfData,rankDocs
import math



# Complete query set with relevant documents
QUERIES = {
    "deep": {
        "query": "deep",
        "relevant": {21, 24, 174, 175, 176, 177, 213, 245, 246, 247, 250, 
                    254, 267, 273, 278, 279, 280, 281, 325, 345, 346, 347, 
                    348, 352, 358, 360, 362, 374, 376, 380, 396, 397, 398, 
                    401, 405, 415, 421, 432}
    },
    "weak heuristic": {
        "query": "weak heuristic",
        "relevant": {1, 35, 93, 101, 172, 174, 213, 257, 299, 306, 361, 391, 413, 429, 435}
    },
    "principle component analysis": {
        "query": "principle component analysis",
        "relevant": {45, 53, 102, 112, 134, 310, 311, 315, 357, 364, 426, 434, 445}
    },
    "human interaction": {
        "query": "human interaction",
        "relevant": {7, 10, 21, 22, 23, 26, 30, 83, 98, 101, 127, 145, 
                    162, 164, 171, 174, 186, 187, 191, 194, 203, 230, 247,
                    249, 250, 255, 256, 265, 273, 289, 345, 369, 383, 391, 
                    395, 403, 426, 428, 436, 444}
    },
    "supervised kernel k-means cluster": {
        "query": "supervised kernel k-means cluster",
        "relevant": {31, 53, 122, 123, 124, 125, 158, 167, 173, 177, 241, 242, 243, 244, 
                    245, 264, 275, 280, 281, 291, 334, 368, 383, 427, 430, 447}
    },
    "patients depression anxiety": {
        "query": "patients depression anxiety",
        "relevant": {37, 40, 62, 72, 80, 168, 225, 259, 263, 328, 332, 333, 355, 368, 391, 
                    400, 433, 447, 448}
    },
    "local global clusters": {
        "query": "local global clusters",
        "relevant": {19, 21, 23, 26, 30, 38, 54, 76, 113, 125, 126, 134, 136, 156, 158, 168, 
                    179, 196, 211, 215, 242, 257, 266, 271, 295, 331, 335, 336, 342, 361, 377, 
                    394, 407, 423}
    },
    "synergy analysis": {
        "query": "synergy analysis",
        "relevant": {38, 102, 112, 134, 315, 357, 434}
    },
    "github mashup apis": {
        "query": "github mashup apis",
        "relevant": {178, 362}
    },
    "Bayesian nonparametric": {
        "query": "Bayesian nonparametric",
        "relevant": {16, 35, 39, 62, 65, 93, 117, 118, 119, 155, 196, 243, 244, 255, 271, 290, 
                    324, 332, 370, 440, 442, 448}
    },
    "diabetes and obesity": {
        "query": "diabetes and obesity",
        "relevant": {72, 148, 391}
    },
    "bootstrap": {
        "query": "bootstrap",
        "relevant": {181, 193, 379}
    },
    "ensemble": {
        "query": "ensemble",
        "relevant": {1, 2, 3, 5, 32, 52, 89, 105, 120, 171, 198, 229, 256, 262, 268, 284, 310, 311, 
                    327, 352, 378, 386, 425}
    },
    "markov": {
        "query": "markov",
        "relevant": {11, 16, 22, 69, 110, 129, 149, 197, 230, 251, 257, 260, 289, 305, 312, 323, 335, 
                    381, 439, 445}
    },
    "prioritize and critical correlate": {
        "query": "prioritize and critical correlate",
        "relevant": {37, 44, 52, 101, 104, 112, 118, 138, 140, 166, 195, 208, 218, 227, 230, 239, 250, 
                    257, 281, 283, 298, 318, 322, 354, 370, 422, 426, 436}
    }
}

def precisionRecall(ranked_docs,relevant_docs):
    retrieved_relevant = 0
    retrieved_docs = set()
    for doc_id,_ in ranked_docs:
        retrieved_docs.add(doc_id)
        if doc_id in relevant_docs:
            retrieved_relevant += 1
    precision = retrieved_relevant/len(retrieved_docs) if retrieved_docs else 0
    recall = retrieved_relevant/len(relevant_docs) if relevant_docs else 0
    return precision, recall

def avgPrecision(ranked_docs,relevant_docs):
    relevant_count = 0
    cumulative_precision = 0.0
    
    for rank,(doc_id,_) in enumerate(ranked_docs,1):
        if doc_id in relevant_docs:
            relevant_count += 1
            cumulative_precision += relevant_count/rank
    
    return cumulative_precision/len(relevant_docs) if relevant_docs else 0

def evalQueries():
    term_list,idf,tfidf_vectors = loadTfidfData()
    evaluation_results = {}
    for query_name,query_data in QUERIES.items():
        ranked_docs = rankDocs(
            query_data["query"],
            term_list,
            idf,
            tfidf_vectors,
            alpha=0.001
        )
        precision,recall = precisionRecall(
            ranked_docs,
            query_data["relevant"]
        )
        avg_precision = avgPrecision(
            ranked_docs,
            query_data["relevant"]
        )
        evaluation_results[query_name] = {
            "precision": precision,
            "recall": recall,
            "average_precision": avg_precision,
            "retrieved": len(ranked_docs),
            "relevant": len(query_data["relevant"])
        }
    return evaluation_results

def meanMetrics(evaluation_results):
    total_queries = len(evaluation_results)
    MAP = sum(result["average_precision"] for result in evaluation_results.values())/total_queries
    MAR = sum(result["recall"] for result in evaluation_results.values())/total_queries
    return MAP,MAR

def showResults(evaluation_results,map_score,mar_score):
    print("\nQuery Evaluation Results:")
    print("=" * 80)
    print(f"{'Query':<30} | {'Precision':<10} | {'Recall':<10} | {'Avg Precision':<12} | {'Retrieved':<10} | {'Relevant':<10}")
    print("=" * 80)
    for query_name,metrics in evaluation_results.items():
        print(f"{query_name[:30]:<30} | {metrics['precision']:<10.4f} | "
              f"{metrics['recall']:<10.4f} | {metrics['average_precision']:<12.4f} | "
              f"{metrics['retrieved']:<10} | {metrics['relevant']:<10}")
    
    print("\nOverall Metrics:")
    print(f"Mean Average Precision (MAP): {map_score:.4f}")
    print(f"Mean Average Recall (MAR): {mar_score:.4f}")


results = evalQueries()
MAP,MAR = meanMetrics(results)
showResults(results,MAP,MAR)