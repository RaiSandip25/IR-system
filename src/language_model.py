import math

class UnigramLanguageModel:    
    def __init__(self, index, mu=2000):
        self.index = index
        self.preprocessor = index.preprocessor
        self.mu = mu
        
        print(f"Unigram Language Model initialized (μ={mu}, using shared index)")
    
    def compute_document_prob(self, term, doc_id):
        # Get term count in document from index
        term_count_doc = self.index.get_term_count_in_doc(term, doc_id)
        
        # Get document length from index
        doc_length = self.index.get_doc_length(doc_id)
        
        # Get collection probability from index
        collection_prob = self.index.get_collection_prob(term)
        
        # Dirichlet smoothing formula
        numerator = term_count_doc + self.mu * collection_prob
        denominator = doc_length + self.mu
        
        return numerator / denominator
    
    def score_document(self, query_terms, doc_id):
        log_likelihood = 0.0
        
        for term in query_terms:
            prob = self.compute_document_prob(term, doc_id)
            
            if prob > 0:
                log_likelihood += math.log(prob)
        
        return log_likelihood
    
    def retrieve(self, query_text, top_k=100):
        # Preprocess query
        query_terms = self.preprocessor.preprocess(query_text)
        
        if not query_terms:
            return []
        
        # Score all documents
        scores = {}
        for doc_id in self.index.documents.keys():
            score = self.score_document(query_terms, doc_id)
            scores[doc_id] = score
        
        # Sort and return top-K
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs[:top_k]
    
    def explain_query(self, query_text, top_n=5):
        print("\n" + "=" * 70)
        print("QUERY EXPLANATION (Language Model)")
        print("=" * 70)
        
        print(f"\nOriginal query: {query_text}")
        
        query_terms = self.preprocessor.preprocess(query_text)
        print(f"Preprocessed terms: {query_terms}")
        
        print(f"\nTerm probabilities in collection:")
        for term in query_terms[:top_n]:
            coll_prob = self.index.get_collection_prob(term)
            coll_count = self.index.get_collection_term_count(term)
            print(f"  '{term}': P(term|collection) = {coll_prob:.6f}")
            print(f"           (appears {coll_count:,} times in collection)")
        
        print(f"\nSmoothing parameter μ = {self.mu}")
        print("=" * 70)
    
    def compare_smoothing(self, query_text, doc_id, mu_values=[500, 1000, 2000, 3000]):
        query_terms = self.preprocessor.preprocess(query_text)
        
        print("\n" + "=" * 70)
        print(f"SMOOTHING COMPARISON (Document {doc_id})")
        print("=" * 70)
        
        print(f"\nQuery: {query_text}")
        print(f"Terms: {query_terms}\n")
        print(f"Document length: {self.index.get_doc_length(doc_id)} terms")
        
        print(f"\n{'μ value':<10} {'Score':<15} {'Effect'}")
        print("-" * 70)
        
        for mu in mu_values:
            original_mu = self.mu
            self.mu = mu
            
            score = self.score_document(query_terms, doc_id)
            
            if mu == min(mu_values):
                effect = "Less smoothing (trust document more)"
            elif mu == max(mu_values):
                effect = "More smoothing (trust collection more)"
            else:
                effect = "Moderate smoothing"
            
            print(f"{mu:<10} {score:<15.4f} {effect}")
            
            self.mu = original_mu
        
        print("=" * 70)