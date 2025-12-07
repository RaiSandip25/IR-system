from data_processing import read_cranfield_data
from preprocessing import TextPreprocessor
from indexer import InvertedIndex
from vsm import VectorSpaceModel
from language_model import UnigramLanguageModel
from evaluation import evaluate_model, calculate_precision_at_k


def save_unified_results(vsm_eval, lm_eval, filename="results/results.txt"):
    with open(filename, 'w') as f:
        # Extract metrics
        vsm_agg = vsm_eval['aggregated']
        lm_agg = lm_eval['aggregated']
        
        # Write in requested format
        f.write("=" * 70 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        # VSM Results
        f.write(f"VSM Precision@5: {vsm_agg['P@5']:.4f}, MAP: {vsm_agg['MAP']:.4f}\n")
        
        # Language Model Results (labeled as Unigram)
        f.write(f"Unigram Precision@5: {lm_agg['P@5']:.4f}, MAP: {lm_agg['MAP']:.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("PERFORMANCE COMPARISON\n")
        f.write("=" * 70 + "\n\n")
        
        # Winner analysis based on MAP
        if lm_agg['MAP'] > vsm_agg['MAP']:
            improvement = ((lm_agg['MAP'] - vsm_agg['MAP']) / vsm_agg['MAP']) * 100
            f.write(f" Unigram Language Model performs better\n")
            f.write(f"  MAP improvement: {improvement:.2f}%\n")
        elif vsm_agg['MAP'] > lm_agg['MAP']:
            improvement = ((vsm_agg['MAP'] - lm_agg['MAP']) / lm_agg['MAP']) * 100
            f.write(f" VSM performs better\n")
            f.write(f"  MAP improvement: {improvement:.2f}%\n")
        else:
            f.write(f" Both models perform equally (MAP)\n")
        
        f.write("\n")
    
    print(f" Results saved to {filename}")


def save_detailed_rankings(vsm_results, lm_results, queries, filename="results/detailed_rankings.txt"):
    with open(filename, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DETAILED QUERY RANKINGS\n")
        f.write("=" * 70 + "\n\n")
        
        for query_id in sorted(queries.keys())[:10]:  # First 10 queries
            query_text = queries[query_id]
            
            f.write(f"Query {query_id}: {query_text[:70]}...\n")
            f.write("-" * 70 + "\n")
            
            # VSM top 10
            f.write("VSM Top 10:\n")
            for rank, (doc_id, score) in enumerate(vsm_results[query_id][:10], 1):
                f.write(f"  {rank:2d}. Doc {doc_id:4d} (score: {score:8.4f})\n")
            
            # LM top 10
            f.write("\nUnigram LM Top 10:\n")
            for rank, (doc_id, score) in enumerate(lm_results[query_id][:10], 1):
                f.write(f"  {rank:2d}. Doc {doc_id:4d} (score: {score:8.4f})\n")
            
            f.write("\n" + "=" * 70 + "\n\n")
    
    print(f" Detailed rankings saved to {filename}")


def print_results_summary(vsm_eval, lm_eval):
    vsm_agg = vsm_eval['aggregated']
    lm_agg = lm_eval['aggregated']
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"VSM Precision@5: {vsm_agg['P@5']:.4f}, MAP: {vsm_agg['MAP']:.4f}")
    print(f"Unigram Precision@5: {lm_agg['P@5']:.4f}, MAP: {lm_agg['MAP']:.4f}")
    print()
    print("=" * 70)


def run_sample_queries(model, model_name, queries, documents, num_samples=3):
    """Run sample queries and display results."""
    print("\n" + "=" * 70)
    print(f"{model_name} - SAMPLE QUERY RESULTS")
    print("=" * 70)
    
    sample_query_ids = list(queries.keys())[:num_samples]
    
    for query_id in sample_query_ids:
        query_text = queries[query_id]
        
        print(f"\n Query {query_id}: {query_text[:80]}...")
        print("-" * 70)
        
        results = model.retrieve(query_text, top_k=5)
        
        print(f"Top 5 Results:")
        for rank, (doc_id, score) in enumerate(results, 1):
            doc_text = documents[doc_id][:100]
            print(f"  {rank}. Doc {doc_id:4d} (score: {score:8.4f})")
            print(f"     {doc_text}...")
    
    print("\n" + "=" * 70)


def run_all_queries(model, queries):
    """Run all queries through the model."""
    results = {}
    for query_id, query_text in queries.items():
        results[query_id] = model.retrieve(query_text, top_k=100)
    return results


def compare_models(vsm_results, lm_results, queries, num_examples=2):
    """Compare results from different models."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    sample_query_ids = list(queries.keys())[:num_examples]
    
    for query_id in sample_query_ids:
        query_text = queries[query_id]
        
        print(f"\n Query {query_id}: {query_text[:80]}...")
        print("-" * 70)
        
        vsm_top5 = vsm_results[query_id][:5]
        lm_top5 = lm_results[query_id][:5]
        
        print(f"\nVSM Top 5:")
        for rank, (doc_id, score) in enumerate(vsm_top5, 1):
            print(f"  {rank}. Doc {doc_id:4d} (score: {score:8.4f})")
        
        print(f"\nUnigram LM Top 5:")
        for rank, (doc_id, score) in enumerate(lm_top5, 1):
            print(f"  {rank}. Doc {doc_id:4d} (score: {score:8.4f})")
        
        # Overlap
        vsm_docs = set(doc_id for doc_id, _ in vsm_top5)
        lm_docs = set(doc_id for doc_id, _ in lm_top5)
        overlap = len(vsm_docs.intersection(lm_docs))
        
        print(f"\n  Overlap: {overlap}/5 documents ({overlap/5*100:.0f}%)")
    
    print("\n" + "=" * 70)


def main():
    """Main execution function."""
    print("=" * 70)
    print("INFORMATION RETRIEVAL SYSTEM")
    print("VSM vs Unigram Language Model")
    print("=" * 70)
    
    # Configuration
    DATA_DIR = "data/cranfield"
    USE_STEMMING = True
    USE_STOPWORDS = True
    DIRICHLET_MU = 2000
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("\n[STEP 1] Loading Data...")
    queries, relevances, documents = read_cranfield_data(DATA_DIR)
    
    # ========================================================================
    # STEP 2: Initialize Preprocessor
    # ========================================================================
    print("\n[STEP 2] Initializing Preprocessor...")
    preprocessor = TextPreprocessor(
        use_stemming=USE_STEMMING,
        use_stopwords=USE_STOPWORDS
    )
    
    # ========================================================================
    # STEP 3: Build Shared Index
    # ========================================================================
    print("\n[STEP 3] Building Shared Inverted Index...")
    index = InvertedIndex(preprocessor)
    index.build_index(documents)
    
    # ========================================================================
    # STEP 4: Initialize Models
    # ========================================================================
    print("\n[STEP 4] Initializing Models...")
    vsm = VectorSpaceModel(index)
    lm = UnigramLanguageModel(index, mu=DIRICHLET_MU)
    
    # ========================================================================
    # STEP 5: Run Sample Queries
    # ========================================================================
    print("\n[STEP 5] Running Sample Queries...")
    run_sample_queries(vsm, "VECTOR SPACE MODEL", queries, documents, num_samples=3)
    run_sample_queries(lm, "UNIGRAM LANGUAGE MODEL", queries, documents, num_samples=3)
    
    # ========================================================================
    # STEP 6: Run All Queries
    # ========================================================================
    print("\n[STEP 6] Processing All Queries...")
    print("  Running VSM on all queries...")
    vsm_results = run_all_queries(vsm, queries)
    
    print("  Running Unigram LM on all queries...")
    lm_results = run_all_queries(lm, queries)
    
    print(f"\n Processed {len(queries)} queries with both models")
    
    # ========================================================================
    # STEP 7: Compare Models
    # ========================================================================
    compare_models(vsm_results, lm_results, queries, num_examples=2)
    
    # ========================================================================
    # STEP 8: Evaluation (Only Precision@5 and MAP)
    # ========================================================================
    print("\n[STEP 8] Running Evaluation (Precision@5 and MAP)...")
    
    vsm_eval = evaluate_model(
        "Vector Space Model (VSM)",
        queries,
        relevances,
        vsm_results,
        k_values=[5]
    )
    
    lm_eval = evaluate_model(
        "Unigram Language Model (Dirichlet)",
        queries,
        relevances,
        lm_results,
        k_values=[5]
    )
    
    # ========================================================================
    # STEP 9: Save Results
    # ========================================================================
    print("\n[STEP 9] Saving Results...")
    
    import os
    os.makedirs("results", exist_ok=True)
    
    # Save unified results file (in requested format)
    save_unified_results(vsm_eval, lm_eval, "results/results.txt")
    
    # Optionally save detailed rankings
    save_detailed_rankings(vsm_results, lm_results, queries, "results/detailed_rankings.txt")
    
    # ========================================================================
    # STEP 10: Display Summary
    # ========================================================================
    print_results_summary(vsm_eval, lm_eval)
    
    print("\n" + "=" * 70)
    print(" SYSTEM EXECUTION COMPLETE")
    print("=" * 70)
    
    print("\nGenerated Files:")
    print(" results/results.txt           - Evaluation summary")
    print(" results/detailed_rankings.txt - Detailed query rankings")
    
    print("\nKey Metrics:")
    print(f"  VSM Precision@5:    {vsm_eval['aggregated']['P@5']:.4f}")
    print(f"  VSM MAP:            {vsm_eval['aggregated']['MAP']:.4f}")
    print(f"  Unigram Precision@5: {lm_eval['aggregated']['P@5']:.4f}")
    print(f"  Unigram MAP:        {lm_eval['aggregated']['MAP']:.4f}")
    
    # Determine winner
    if lm_eval['aggregated']['MAP'] > vsm_eval['aggregated']['MAP']:
        improvement = ((lm_eval['aggregated']['MAP'] - vsm_eval['aggregated']['MAP']) / 
                      vsm_eval['aggregated']['MAP']) * 100
        print(f"\n  Winner: Unigram Language Model ({improvement:.2f}% better)")
    elif vsm_eval['aggregated']['MAP'] > lm_eval['aggregated']['MAP']:
        improvement = ((vsm_eval['aggregated']['MAP'] - lm_eval['aggregated']['MAP']) / 
                      lm_eval['aggregated']['MAP']) * 100
        print(f"\n  Winner: VSM ({improvement:.2f}% better)")
    else:
        print(f"\n  Tie: Both models perform equally")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print("\nPlease ensure Cranfield dataset is in data/cranfield/")
        print("Required files:")
        print("  - cran.all.1400")
        print("  - cran.qry")
        print("  - cranqrel")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback
        traceback.print_exc()