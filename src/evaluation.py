def calculate_precision_at_k(relevant_docs, ranked_docs, k):
    if not ranked_docs or k == 0:
        return 0.0
    
    # Get top-K documents
    top_k = ranked_docs[:k]
    relevant_set = set(relevant_docs)
    
    # Count relevant documents in top-K
    relevant_retrieved = sum(1 for doc_id, _ in top_k if doc_id in relevant_set)
    
    return relevant_retrieved / k


def calculate_average_precision(relevant_docs, ranked_docs):
    if not relevant_docs:
        return 0.0
    
    relevant_set = set(relevant_docs)
    num_relevant = len(relevant_docs)
    
    score = 0.0
    num_hits = 0
    
    # Iterate through ranked documents
    for i, (doc_id, _) in enumerate(ranked_docs, 1):
        if doc_id in relevant_set:
            num_hits += 1
            # Precision at this position
            precision_at_i = num_hits / i
            score += precision_at_i
    
    # Average over all relevant documents
    return score / num_relevant if num_relevant > 0 else 0.0


def calculate_map(queries, relevances, results):
    average_precisions = []
    
    for query_id in queries.keys():
        relevant_docs = relevances.get(query_id, [])
        ranked_docs = results.get(query_id, [])
        
        if relevant_docs:  # Only consider queries with relevance judgments
            ap = calculate_average_precision(relevant_docs, ranked_docs)
            average_precisions.append(ap)
    
    if not average_precisions:
        return 0.0
    
    return sum(average_precisions) / len(average_precisions)


def evaluate_model(model_name, queries, relevances, results, k_values=[5]):
    print("\n" + "=" * 70)
    print(f"EVALUATING: {model_name}")
    print("=" * 70)
    
    metrics = {
        'precision_at_k': {},
        'average_precisions': []
    }
    
    # Per-query metrics
    for query_id in queries.keys():
        relevant_docs = relevances.get(query_id, [])
        ranked_docs = results.get(query_id, [])
        
        if not relevant_docs:
            continue
        
        # Calculate Precision@K
        for k in k_values:
            if k not in metrics['precision_at_k']:
                metrics['precision_at_k'][k] = []
            
            metrics['precision_at_k'][k].append(
                calculate_precision_at_k(relevant_docs, ranked_docs, k)
            )
        
        # Average Precision
        metrics['average_precisions'].append(
            calculate_average_precision(relevant_docs, ranked_docs)
        )
    
    # Aggregate metrics
    aggregated = {
        'MAP': sum(metrics['average_precisions']) / len(metrics['average_precisions'])
    }
    
    for k in k_values:
        aggregated[f'P@{k}'] = sum(metrics['precision_at_k'][k]) / len(metrics['precision_at_k'][k])
    
    # Print results
    print(f"\nResults:")
    print(f"  Precision@5: {aggregated['P@5']:.4f}")
    print(f"  MAP:         {aggregated['MAP']:.4f}")
    print("=" * 70)
    
    return {
        'raw_metrics': metrics,
        'aggregated': aggregated
    }