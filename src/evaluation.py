import math


def calculate_precision_at_k(relevant_docs, ranked_docs, k):
  
    if not ranked_docs or k == 0:
        return 0.0
    
    top_k = ranked_docs[:k]
    relevant_set = set(relevant_docs)
    relevant_retrieved = sum(1 for doc_id, _ in top_k if doc_id in relevant_set)
    
    return relevant_retrieved / k


def calculate_recall_at_k(relevant_docs, ranked_docs, k):
  
    if not relevant_docs:
        return 0.0
    
    top_k = ranked_docs[:k]
    relevant_set = set(relevant_docs)
    relevant_retrieved = sum(1 for doc_id, _ in top_k if doc_id in relevant_set)
    
    return relevant_retrieved / len(relevant_docs)


def calculate_f1_at_k(relevant_docs, ranked_docs, k):
   
    precision = calculate_precision_at_k(relevant_docs, ranked_docs, k)
    recall = calculate_recall_at_k(relevant_docs, ranked_docs, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def calculate_average_precision(relevant_docs, ranked_docs):

    if not relevant_docs:
        return 0.0
    
    relevant_set = set(relevant_docs)
    num_relevant = len(relevant_docs)
    
    score = 0.0
    num_hits = 0
    
    for i, (doc_id, _) in enumerate(ranked_docs, 1):
        if doc_id in relevant_set:
            num_hits += 1
            precision_at_i = num_hits / i
            score += precision_at_i
    
    return score / num_relevant if num_relevant > 0 else 0.0


def calculate_dcg_at_k(relevant_docs, ranked_docs, k):
    
    if not ranked_docs or k == 0:
        return 0.0
    
    relevant_set = set(relevant_docs)
    dcg = 0.0
    
    for i, (doc_id, _) in enumerate(ranked_docs[:k], 1):
        # Binary relevance: 1 if relevant, 0 otherwise
        rel = 1 if doc_id in relevant_set else 0
        dcg += rel / math.log2(i + 1)
    
    return dcg


def calculate_idcg_at_k(relevant_docs, k):
  
    if not relevant_docs:
        return 0.0
    
    # For binary relevance, ideal ranking has all relevant docs first
    num_relevant = min(len(relevant_docs), k)
    idcg = 0.0
    
    for i in range(1, num_relevant + 1):
        idcg += 1.0 / math.log2(i + 1)
    
    return idcg


def calculate_ndcg_at_k(relevant_docs, ranked_docs, k):
  
    dcg = calculate_dcg_at_k(relevant_docs, ranked_docs, k)
    idcg = calculate_idcg_at_k(relevant_docs, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_reciprocal_rank(relevant_docs, ranked_docs):
  
    if not relevant_docs:
        return 0.0
    
    relevant_set = set(relevant_docs)
    
    for i, (doc_id, _) in enumerate(ranked_docs, 1):
        if doc_id in relevant_set:
            return 1.0 / i
    
    return 0.0


def calculate_r_precision(relevant_docs, ranked_docs):
   
    if not relevant_docs:
        return 0.0
    
    r = len(relevant_docs)
    return calculate_precision_at_k(relevant_docs, ranked_docs, r)


def calculate_err_at_k(relevant_docs, ranked_docs, k, max_grade=1):
  
    if not ranked_docs or k == 0:
        return 0.0
    
    relevant_set = set(relevant_docs)
    err = 0.0
    p = 1.0  # Probability user hasn't found a relevant doc yet
    
    for i, (doc_id, _) in enumerate(ranked_docs[:k], 1):
        # Binary relevance: 1 if relevant, 0 otherwise
        grade = 1 if doc_id in relevant_set else 0
        
        # Relevance probability (utility)
        # For binary: R = grade / max_grade
        r = grade / max_grade
        
        # Contribution of this position
        err += p * r / i
        
        # Update probability of continuing (user hasn't stopped yet)
        p *= (1 - r)
    
    return err


def calculate_map(queries, relevances, results):
    """Calculate Mean Average Precision (MAP)."""
    average_precisions = []
    
    for query_id in queries.keys():
        relevant_docs = relevances.get(query_id, [])
        ranked_docs = results.get(query_id, [])
        
        if relevant_docs:
            ap = calculate_average_precision(relevant_docs, ranked_docs)
            average_precisions.append(ap)
    
    if not average_precisions:
        return 0.0
    
    return sum(average_precisions) / len(average_precisions)


def calculate_mrr(queries, relevances, results):
    """Calculate Mean Reciprocal Rank (MRR)."""
    reciprocal_ranks = []
    
    for query_id in queries.keys():
        relevant_docs = relevances.get(query_id, [])
        ranked_docs = results.get(query_id, [])
        
        if relevant_docs:
            rr = calculate_reciprocal_rank(relevant_docs, ranked_docs)
            reciprocal_ranks.append(rr)
    
    if not reciprocal_ranks:
        return 0.0
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def evaluate_model(model_name, queries, relevances, results, k_values=[5, 10]):

    print("\n" + "=" * 70)
    print(f"EVALUATING: {model_name}")
    print("=" * 70)
    
    metrics = {
        'precision_at_k': {k: [] for k in k_values},
        'recall_at_k': {k: [] for k in k_values},
        'f1_at_k': {k: [] for k in k_values},
        'ndcg_at_k': {k: [] for k in k_values},
        'err_at_k': {k: [] for k in k_values},
        'average_precisions': [],
        'r_precisions': []
    }
    
    # Per-query metrics
    num_queries_evaluated = 0
    for query_id in queries.keys():
        relevant_docs = relevances.get(query_id, [])
        ranked_docs = results.get(query_id, [])
        
        if not relevant_docs:
            continue
        
        num_queries_evaluated += 1
        
        # Calculate all metrics at different K values
        for k in k_values:
            metrics['precision_at_k'][k].append(
                calculate_precision_at_k(relevant_docs, ranked_docs, k)
            )
            metrics['recall_at_k'][k].append(
                calculate_recall_at_k(relevant_docs, ranked_docs, k)
            )
            metrics['f1_at_k'][k].append(
                calculate_f1_at_k(relevant_docs, ranked_docs, k)
            )
            metrics['ndcg_at_k'][k].append(
                calculate_ndcg_at_k(relevant_docs, ranked_docs, k)
            )
            metrics['err_at_k'][k].append(
                calculate_err_at_k(relevant_docs, ranked_docs, k)
            )
        
        # Single-value metrics
        metrics['average_precisions'].append(
            calculate_average_precision(relevant_docs, ranked_docs)
        )
        metrics['r_precisions'].append(
            calculate_r_precision(relevant_docs, ranked_docs)
        )
    
    # Aggregate metrics
    aggregated = {
        'MAP': sum(metrics['average_precisions']) / len(metrics['average_precisions']),
        'R-Precision': sum(metrics['r_precisions']) / len(metrics['r_precisions'])
    }
    
    for k in k_values:
        aggregated[f'P@{k}'] = sum(metrics['precision_at_k'][k]) / len(metrics['precision_at_k'][k])
        aggregated[f'R@{k}'] = sum(metrics['recall_at_k'][k]) / len(metrics['recall_at_k'][k])
        aggregated[f'F1@{k}'] = sum(metrics['f1_at_k'][k]) / len(metrics['f1_at_k'][k])
        aggregated[f'nDCG@{k}'] = sum(metrics['ndcg_at_k'][k]) / len(metrics['ndcg_at_k'][k])
        aggregated[f'ERR@{k}'] = sum(metrics['err_at_k'][k]) / len(metrics['err_at_k'][k])
    
    # Print results
    print(f"\nResults ({num_queries_evaluated} queries evaluated):")
    print(f"\n  Core Metrics:")
    print(f"    MAP:           {aggregated['MAP']:.4f}")
    print(f"    R-Precision:   {aggregated['R-Precision']:.4f}")
    
    for k in k_values:
        print(f"\n  Metrics @ K={k}:")
        print(f"    Precision@{k}:   {aggregated[f'P@{k}']:.4f}")
        print(f"    Recall@{k}:      {aggregated[f'R@{k}']:.4f}")
        print(f"    F1@{k}:          {aggregated[f'F1@{k}']:.4f}")
        print(f"    nDCG@{k}:        {aggregated[f'nDCG@{k}']:.4f}")
        print(f"    ERR@{k}:         {aggregated[f'ERR@{k}']:.4f}")
    
    print("=" * 70)
    
    return {
        'raw_metrics': metrics,
        'aggregated': aggregated,
        'num_queries': num_queries_evaluated
    }