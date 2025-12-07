import math
from collections import Counter, defaultdict


class InvertedIndex:    
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        
        # Core index structure
        self.index = defaultdict(list)  # term → [(doc_id, freq), ...]
        
        # Document statistics
        self.documents = {}  # {doc_id: document_text}
        self.doc_term_counts = {}  # {doc_id: {term: count}}
        self.doc_lengths = {}  # {doc_id: total number of terms}
        
        # Collection statistics
        self.doc_freq = {}  # {term: number of docs containing term}
        self.idf = {}  # {term: IDF value}
        self.vocabulary = set()  # All unique terms
        self.num_docs = 0
        self.total_terms = 0  # Total terms in collection
        self.avg_doc_length = 0.0
        
        # Collection-wide term counts (for language models)
        self.collection_term_counts = Counter()  # {term: total count in collection}
        
        print("Inverted Index initialized")
    
    def build_index(self, documents):
        print("\n" + "=" * 70)
        print("BUILDING INVERTED INDEX")
        print("=" * 70)
        
        self.documents = documents
        self.num_docs = len(documents)

        print("\nStep 1: Processing documents and building index...")
        
        for doc_id, text in documents.items():
            # Preprocess document
            tokens = self.preprocessor.preprocess(text)
            
            # Count term frequencies in this document
            term_counts = Counter(tokens)
            self.doc_term_counts[doc_id] = term_counts
            
            # Document length
            doc_length = len(tokens)
            self.doc_lengths[doc_id] = doc_length
            self.total_terms += doc_length
            
            # Update vocabulary
            self.vocabulary.update(tokens)
            
            # Update collection term counts (for language models)
            self.collection_term_counts.update(tokens)
            
            # Build inverted index
            for term, count in term_counts.items():
                self.index[term].append((doc_id, count))
        
        print("Step 2: Computing document frequencies...")
        
        for term, postings in self.index.items():
            # Document frequency = number of documents containing this term
            self.doc_freq[term] = len(postings)

        print("Step 3: Computing collection statistics...")
        
        self.avg_doc_length = self.total_terms / self.num_docs if self.num_docs > 0 else 0

        print("Step 4: Computing IDF values...")
        
        self.compute_idf()

        print("\n✓ Index built successfully!")
        print(f"\nIndex Statistics:")
        print(f"  Documents indexed:        {self.num_docs:,}")
        print(f"  Vocabulary size:          {len(self.vocabulary):,} unique terms")
        print(f"  Total terms in collection: {self.total_terms:,}")
        print(f"  Average document length:   {self.avg_doc_length:.1f} terms")
        print(f"  Average postings per term: {sum(len(p) for p in self.index.values())/len(self.index):.1f}")
        
        # Show sample terms
        print(f"\nSample Index Entries:")
        sample_terms = sorted(self.index.keys())[:5]
        for term in sample_terms:
            postings = self.index[term]
            df = self.doc_freq[term]
            idf = self.idf[term]
            print(f"  '{term}': appears in {df} docs, IDF={idf:.4f}, {len(postings)} postings")
        
        print("=" * 70)
    
    def compute_idf(self):
        for term, df in self.doc_freq.items():
            if df > 0:
                # Standard IDF formula
                self.idf[term] = math.log(self.num_docs / df)
            else:
                self.idf[term] = 0.0
    
    def get_postings(self, term):
        
        return self.index.get(term, [])
    
    def get_doc_freq(self, term):

        return self.doc_freq.get(term, 0)
    
    def get_idf(self, term):
 
        return self.idf.get(term, 0.0)
    
    def get_term_count_in_doc(self, term, doc_id):

        return self.doc_term_counts.get(doc_id, {}).get(term, 0)
    
    def get_doc_length(self, doc_id):

        return self.doc_lengths.get(doc_id, 0)
    
    def get_collection_term_count(self, term):

        return self.collection_term_counts.get(term, 0)
    
    def get_collection_prob(self, term):
        
        if self.total_terms == 0:
            return 0.0
        return self.collection_term_counts.get(term, 0) / self.total_terms
    
    def term_exists(self, term):

        return term in self.vocabulary
    
    def get_documents_containing_term(self, term):

        return [doc_id for doc_id, _ in self.get_postings(term)]
    
    def print_statistics(self):

        print("\n" + "=" * 70)
        print("INVERTED INDEX STATISTICS")
        print("=" * 70)
        
        print(f"\nDocument Statistics:")
        print(f"  Total documents:          {self.num_docs:,}")
        print(f"  Average document length:   {self.avg_doc_length:.1f} terms")
        print(f"  Shortest document:         {min(self.doc_lengths.values()):,} terms")
        print(f"  Longest document:          {max(self.doc_lengths.values()):,} terms")
        
        print(f"\nVocabulary Statistics:")
        print(f"  Vocabulary size:           {len(self.vocabulary):,} unique terms")
        print(f"  Total terms in collection: {self.total_terms:,}")
        
        print(f"\nIndex Statistics:")
        print(f"  Index entries:             {len(self.index):,}")
        print(f"  Total postings:            {sum(len(p) for p in self.index.values()):,}")
        print(f"  Avg postings per term:     {sum(len(p) for p in self.index.values())/len(self.index):.1f}")
        
        # Most/least common terms
        sorted_by_df = sorted(self.doc_freq.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nMost common terms (by document frequency):")
        for term, df in sorted_by_df[:5]:
            print(f"  '{term}': appears in {df} docs ({df/self.num_docs*100:.1f}%)")
        
        print(f"\nLeast common terms (by document frequency):")
        for term, df in sorted_by_df[-5:]:
            print(f"  '{term}': appears in {df} docs ({df/self.num_docs*100:.1f}%)")
        
        print("=" * 70)
    
    def search_term(self, term):

        print("\n" + "=" * 70)
        print(f"TERM SEARCH: '{term}'")
        print("=" * 70)
        
        # Preprocess the term
        processed_terms = self.preprocessor.preprocess(term)
        
        if not processed_terms:
            print(f"\nNo valid terms after preprocessing")
            return
        
        processed_term = processed_terms[0]
        print(f"\nPreprocessed term: '{processed_term}'")
        
        if processed_term not in self.vocabulary:
            print(f"\nTerm not found in vocabulary")
            return
        
        # Get statistics
        postings = self.get_postings(processed_term)
        df = self.get_doc_freq(processed_term)
        idf = self.get_idf(processed_term)
        collection_count = self.get_collection_term_count(processed_term)
        collection_prob = self.get_collection_prob(processed_term)
        
        print(f"\nTerm Statistics:")
        print(f"  Document frequency (df):    {df} documents ({df/self.num_docs*100:.2f}%)")
        print(f"  IDF value:                  {idf:.4f}")
        print(f"  Total occurrences:          {collection_count:,}")
        print(f"  Collection probability:     {collection_prob:.6f}")
        
        print(f"\nSample Documents (top 5 by frequency):")
        sorted_postings = sorted(postings, key=lambda x: x[1], reverse=True)[:5]
        for doc_id, freq in sorted_postings:
            doc_length = self.get_doc_length(doc_id)
            print(f"  Doc {doc_id:4d}: {freq:3d} occurrences (doc length: {doc_length:4d} terms)")
        
        print("=" * 70)