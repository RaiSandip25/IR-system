import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    
    def __init__(self, use_stemming=True, use_stopwords=True):
        self.use_stemming = use_stemming
        self.use_stopwords = use_stopwords
        
        # Initialize Porter Stemmer
        if use_stemming:
            self.stemmer = PorterStemmer()
        else:
            self.stemmer = None
        
        # Load English stopwords
        if use_stopwords:
            self.stopwords = set(stopwords.words('english'))
            # You can add custom stopwords here if needed
            # self.stopwords.update(['custom', 'words'])
        else:
            self.stopwords = set()
        
        print(f"TextPreprocessor initialized:")
        print(f"  - Stemming: {'ON' if use_stemming else 'OFF'}")
        print(f"  - Stopwords: {'ON' if use_stopwords else 'OFF'} ({len(self.stopwords)} words)")
    
    def tokenize(self, text):
        # Convert to lowercase and extract alphabetic words
        # Pattern: \b[a-z]+\b matches word boundaries with alphabetic chars
        tokens = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Filter out single-character tokens (optional, but common in IR)
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def remove_stopwords(self, tokens):
        
        return [token for token in tokens if token not in self.stopwords]
    
    def stem_tokens(self, tokens):
        
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess(self, text):

        if not text or not isinstance(text, str):
            return []
        
        # Step 1: Tokenize (lowercase + extract words)
        tokens = self.tokenize(text)
        
        # Step 2: Remove stopwords (if enabled)
        if self.use_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Step 3: Stem (if enabled)
        if self.use_stemming:
            tokens = self.stem_tokens(tokens)
        
        return tokens
    
    def get_term_frequencies(self, text):

        tokens = self.preprocess(text)
        return Counter(tokens)
    
    def preprocess_batch(self, texts):

        return [self.preprocess(text) for text in texts]


def compare_preprocessing(text, stemming=True, stopwords=True):

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPARISON")
    print("=" * 70)
    
    print(f"\nOriginal text:")
    print(f"  {text}")
    
    # No preprocessing
    print(f"\n1. Tokenized only:")
    preprocessor = TextPreprocessor(use_stemming=False, use_stopwords=False)
    tokens = preprocessor.preprocess(text)
    print(f"  {tokens}")
    print(f"  Token count: {len(tokens)}")
    
    # With stopword removal
    print(f"\n2. Tokenized + Stopword removal:")
    preprocessor = TextPreprocessor(use_stemming=False, use_stopwords=True)
    tokens = preprocessor.preprocess(text)
    print(f"  {tokens}")
    print(f"  Token count: {len(tokens)}")
    
    # With stemming
    print(f"\n3. Tokenized + Stemming:")
    preprocessor = TextPreprocessor(use_stemming=True, use_stopwords=False)
    tokens = preprocessor.preprocess(text)
    print(f"  {tokens}")
    print(f"  Token count: {len(tokens)}")
    
    # Full preprocessing
    print(f"\n4. Full preprocessing (Stopwords + Stemming):")
    preprocessor = TextPreprocessor(use_stemming=True, use_stopwords=True)
    tokens = preprocessor.preprocess(text)
    print(f"  {tokens}")
    print(f"  Token count: {len(tokens)}")
    
    print("\n" + "=" * 70)
