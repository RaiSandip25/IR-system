import math
from collections import Counter


class VectorSpaceModel:
    def __init__(self, index):
        self.index = index
        self.preprocessor = index.preprocessor
        
        print("Vector Space Model initialized (using shared index)")
    
    def compute_tf(self, term_freq, doc_length):
        if doc_length == 0:
            return 0.0
        return term_freq / doc_length
    
    def compute_tfidf(self, term, doc_id):
        # Get term frequency from index
        term_freq = self.index.get_term_count_in_doc(term, doc_id)
        
        if term_freq == 0:
            return 0.0
        
        # Get document length from index
        doc_length = self.index.get_doc_length(doc_id)
        
        # Compute TF
        tf = self.compute_tf(term_freq, doc_length)
        
        # Get IDF from index
        idf = self.index.get_idf(term)
        
        # TF-IDF
        return tf * idf
    
    def get_document_vector(self, doc_id):
        doc_vector = {}
        
        # Get terms in this document from index
        term_counts = self.index.doc_term_counts.get(doc_id, {})
        
        # Compute TF-IDF for each term
        for term in term_counts.keys():
            doc_vector[term] = self.compute_tfidf(term, doc_id)
        
        return doc_vector
    
    def get_query_vector(self, query_text):
        # Preprocess query
        query_terms = self.preprocessor.preprocess(query_text)
        query_term_freqs = Counter(query_terms)
        query_length = len(query_terms)
        
        query_vector = {}
        
        # Compute TF-IDF for query terms
        for term, freq in query_term_freqs.items():
            if self.index.term_exists(term):  # Only use terms in vocabulary
                tf = freq / query_length if query_length > 0 else 0
                idf = self.index.get_idf(term)
                query_vector[term] = tf * idf
        
        return query_vector
    
    def cosine_similarity(self, vec1, vec2):
        # Get common terms
        common_terms = set(vec1.keys()).intersection(set(vec2.keys()))
        
        if not common_terms:
            return 0.0
        
        # Compute dot product
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # Compute magnitudes
        magnitude1 = math.sqrt(sum(weight ** 2 for weight in vec1.values()))
        magnitude2 = math.sqrt(sum(weight ** 2 for weight in vec2.values()))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def retrieve(self, query_text, top_k=100):
        # Get query vector
        query_vector = self.get_query_vector(query_text)
        
        if not query_vector:
            return []
        
        # Compute similarity with all documents
        scores = {}
        for doc_id in self.index.documents.keys():
            doc_vector = self.get_document_vector(doc_id)
            similarity = self.cosine_similarity(query_vector, doc_vector)
            
            if similarity > 0:
                scores[doc_id] = similarity
        
        # Sort and return top-K
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs[:top_k]
    
    def explain_query(self, query_text, top_n=5):
        """Explain query processing."""
        print("\n" + "=" * 70)
        print("QUERY EXPLANATION (VSM)")
        print("=" * 70)
        
        print(f"\nOriginal query: {query_text}")
        
        query_terms = self.preprocessor.preprocess(query_text)
        print(f"Preprocessed terms: {query_terms}")
        
        query_vector = self.get_query_vector(query_text)
        
        print(f"\nQuery TF-IDF weights (top {top_n}):")
        sorted_terms = sorted(query_vector.items(), key=lambda x: x[1], reverse=True)
        for term, weight in sorted_terms[:top_n]:
            df = self.index.get_doc_freq(term)
            idf = self.index.get_idf(term)
            print(f"  '{term}': TF-IDF={weight:.4f} (df={df}, IDF={idf:.4f})")
        
        print("=" * 70)