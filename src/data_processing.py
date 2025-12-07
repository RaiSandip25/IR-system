import os

def parse_cranfield_documents(file_path):
    documents = {}
    current_doc_id = None
    current_field = None
    current_text = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            # Document ID marker
            if line.startswith('.I'):
                # Save previous document
                if current_doc_id is not None:
                    documents[current_doc_id] = ' '.join(current_text).strip()
                
                # Start new document
                current_doc_id = int(line.split()[1])
                current_text = []
                current_field = None
            
            # Field markers
            elif line.startswith('.T'):
                current_field = 'title'
            elif line.startswith('.A'):
                current_field = 'author'
            elif line.startswith('.B'):
                current_field = 'bibliography'
            elif line.startswith('.W'):
                current_field = 'abstract'
            
            # Content lines - we only use title and abstract for retrieval
            elif current_field in ['title', 'abstract']:
                if line:  # Skip empty lines
                    current_text.append(line)
        
        # Save last document
        if current_doc_id is not None:
            documents[current_doc_id] = ' '.join(current_text).strip()
    
    return documents


def parse_cranfield_queries(file_path):

    queries = {}
    current_query_id = None
    current_text = []
    in_query_text = False
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            # Query ID marker
            if line.startswith('.I'):
                # Save previous query
                if current_query_id is not None:
                    queries[current_query_id] = ' '.join(current_text).strip()
                
                # Start new query
                current_query_id = int(line.split()[1])
                current_text = []
                in_query_text = False
            
            # Query text marker
            elif line.startswith('.W'):
                in_query_text = True
            
            # Content lines
            elif in_query_text and line:
                current_text.append(line)
        
        # Save last query
        if current_query_id is not None:
            queries[current_query_id] = ' '.join(current_text).strip()
    
    return queries


def parse_cranfield_relevance(file_path):
    relevances = {}
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                query_id = int(parts[0])
                doc_id = int(parts[1])
                # relevance_score = int(parts[2]) if len(parts) >= 3 else 1
                
                # Add document to relevant list for this query
                if query_id not in relevances:
                    relevances[query_id] = []
                relevances[query_id].append(doc_id)
    
    return relevances


def read_cranfield_data(data_dir):
    # File paths
    doc_file = os.path.join(data_dir, 'cran.all.1400')
    query_file = os.path.join(data_dir, 'cran.qry')
    rel_file = os.path.join(data_dir, 'cranqrel')
    
    # Check files exist
    for filepath in [doc_file, query_file, rel_file]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required file not found: {filepath}")
    
    print("=" * 70)
    print("Reading Cranfield dataset...")
    print("=" * 70)
    
    # Parse files
    documents = parse_cranfield_documents(doc_file)
    queries = parse_cranfield_queries(query_file)
    relevances = parse_cranfield_relevance(rel_file)
    
    # Print statistics
    print(f"✓ Read {len(documents)} documents")
    print(f"✓ Read {len(queries)} queries")
    print(f"✓ Read relevance judgments for {len(relevances)} queries")
    
    # Calculate average document length
    avg_doc_length = sum(len(doc.split()) for doc in documents.values()) / len(documents)
    avg_query_length = sum(len(q.split()) for q in queries.values()) / len(queries)
    
    print(f"\nDataset Statistics:")
    print(f"  Average document length: {avg_doc_length:.1f} words")
    print(f"  Average query length: {avg_query_length:.1f} words")
    
    # Show sample
    sample_doc_id = list(documents.keys())[0]
    sample_query_id = list(queries.keys())[0]
    
    print(f"\nSample Document (ID {sample_doc_id}):")
    print(f"  {documents[sample_doc_id][:100]}...")
    
    print(f"\nSample Query (ID {sample_query_id}):")
    print(f"  {queries[sample_query_id][:100]}...")
    
    print("=" * 70)
    
    return queries, relevances, documents