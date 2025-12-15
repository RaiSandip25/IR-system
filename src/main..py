from data_processing import read_cranfield_data
from preprocessing import TextPreprocessor
from indexer import InvertedIndex
from vsm import VectorSpaceModel
from language_model import UnigramLanguageModel
from evaluation import evaluate_model


def save_unified_results(vsm_eval, lm_eval, filename="results/results.txt"):
    """Save results in table format with all metrics."""
    with open(filename, 'w') as f:
        vsm_agg = vsm_eval['aggregated']
        lm_agg = lm_eval['aggregated']
        
        f.write("=" * 110 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 110 + "\n\n")
        
        # Table header
        f.write(f"{'Model':<15} | {'R@5':<8} | {'P@5':<8} | {'F1@5':<8} | {'R@10':<8} | {'P@10':<8} | {'F1@10':<8} | {'MAP':<8} | {'nDCG@10':<8} | {'ERR@10':<8}\n")
        f.write("-" * 110 + "\n")
        
        # VSM row
        f.write(f"{'VSM':<15} | {vsm_agg['R@5']:<8.4f} | {vsm_agg['P@5']:<8.4f} | {vsm_agg['F1@5']:<8.4f} | "
                f"{vsm_agg['R@10']:<8.4f} | {vsm_agg['P@10']:<8.4f} | {vsm_agg['F1@10']:<8.4f} | "
                f"{vsm_agg['MAP']:<8.4f} | {vsm_agg['nDCG@10']:<8.4f} | {vsm_agg['ERR@10']:<8.4f}\n")
        f.write("=" * 110 + "\n")
        
        # Dirichlet LM row
        f.write(f"{'Dirichlet LM':<15} | {lm_agg['R@5']:<8.4f} | {lm_agg['P@5']:<8.4f} | {lm_agg['F1@5']:<8.4f} | "
                f"{lm_agg['R@10']:<8.4f} | {lm_agg['P@10']:<8.4f} | {lm_agg['F1@10']:<8.4f} | "
                f"{lm_agg['MAP']:<8.4f} | {lm_agg['nDCG@10']:<8.4f} | {lm_agg['ERR@10']:<8.4f}\n")
        f.write("=" * 110 + "\n\n")
        
        # Performance comparison
        f.write("PERFORMANCE COMPARISON\n")
        f.write("=" * 110 + "\n\n")
        
        # Winner analysis based on MAP
        if lm_agg['MAP'] > vsm_agg['MAP']:
            improvement = ((lm_agg['MAP'] - vsm_agg['MAP']) / vsm_agg['MAP']) * 100
            f.write(f"Winner: Dirichlet LM performs better\n")
            f.write(f"  MAP improvement: {improvement:.2f}%\n")
        elif vsm_agg['MAP'] > lm_agg['MAP']:
            improvement = ((vsm_agg['MAP'] - lm_agg['MAP']) / lm_agg['MAP']) * 100
            f.write(f"Winner: VSM performs better\n")
            f.write(f"  MAP improvement: {improvement:.2f}%\n")
        else:
            f.write(f"Result: Both models perform equally (MAP)\n")
        
        f.write("\n")
    
    print(f"✓ Results saved to {filename}")


def save_comprehensive_metrics(vsm_eval, lm_eval, filename="results/metrics.txt"):
    with open(filename, 'w') as f:
        vsm_agg = vsm_eval['aggregated']
        lm_agg = lm_eval['aggregated']
        
        f.write("=" * 70 + "\n")
        f.write("COMPREHENSIVE EVALUATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        # Summary metrics (original format for compatibility)
        f.write("Quick Summary:\n")
        f.write(f"VSM Precision@5: {vsm_agg['P@5']:.4f}, MAP: {vsm_agg['MAP']:.4f}\n")
        f.write(f"Unigram Precision@5: {lm_agg['P@5']:.4f}, MAP: {lm_agg['MAP']:.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("DETAILED METRICS\n")
        f.write("=" * 70 + "\n\n")
        
        # VSM Detailed Results
        f.write("Vector Space Model (VSM):\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Core Metrics:\n")
        f.write(f"    MAP:           {vsm_agg['MAP']:.4f}\n")
        f.write(f"    MRR:           {vsm_agg['MRR']:.4f}\n")
        f.write(f"    R-Precision:   {vsm_agg['R-Precision']:.4f}\n\n")
        
        for k in [5, 10, 20]:
            f.write(f"  Metrics @ K={k}:\n")
            f.write(f"    Precision@{k}:   {vsm_agg[f'P@{k}']:.4f}\n")
            f.write(f"    Recall@{k}:      {vsm_agg[f'R@{k}']:.4f}\n")
            f.write(f"    F1@{k}:          {vsm_agg[f'F1@{k}']:.4f}\n")
            f.write(f"    nDCG@{k}:        {vsm_agg[f'nDCG@{k}']:.4f}\n\n")
        
        # Unigram LM Detailed Results
        f.write("\nUnigram Language Model:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Core Metrics:\n")
        f.write(f"    MAP:           {lm_agg['MAP']:.4f}\n")
        f.write(f"    MRR:           {lm_agg['MRR']:.4f}\n")
        f.write(f"    R-Precision:   {lm_agg['R-Precision']:.4f}\n\n")
        
        for k in [5, 10, 20]:
            f.write(f"  Metrics @ K={k}:\n")
            f.write(f"    Precision@{k}:   {lm_agg[f'P@{k}']:.4f}\n")
            f.write(f"    Recall@{k}:      {lm_agg[f'R@{k}']:.4f}\n")
            f.write(f"    F1@{k}:          {lm_agg[f'F1@{k}']:.4f}\n")
            f.write(f"    nDCG@{k}:        {lm_agg[f'nDCG@{k}']:.4f}\n\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("PERFORMANCE COMPARISON\n")
        f.write("=" * 70 + "\n\n")
        
        # Compare across multiple metrics
        comparisons = [
            ('MAP', 'MAP'),
            ('MRR', 'MRR'),
            ('nDCG@10', 'nDCG@10'),
            ('Precision@5', 'P@5'),
            ('Recall@10', 'R@10')
        ]
        
        for metric_name, metric_key in comparisons:
            vsm_val = vsm_agg[metric_key]
            lm_val = lm_agg[metric_key]
            
            if lm_val > vsm_val:
                improvement = ((lm_val - vsm_val) / vsm_val) * 100 if vsm_val > 0 else 0
                winner = "Unigram LM"
            elif vsm_val > lm_val:
                improvement = ((vsm_val - lm_val) / lm_val) * 100 if lm_val > 0 else 0
                winner = "VSM"
            else:
                improvement = 0
                winner = "Tie"
            
            f.write(f"  {metric_name}:\n")
            f.write(f"    VSM: {vsm_val:.4f}, Unigram LM: {lm_val:.4f}\n")
            if winner != "Tie":
                f.write(f"    Winner: {winner} (+{improvement:.2f}%)\n\n")
            else:
                f.write(f"    Result: Tie\n\n")
        
        # Overall winner based on MAP
        f.write("-" * 70 + "\n")
        f.write("Overall Winner (based on MAP):\n")
        if lm_agg['MAP'] > vsm_agg['MAP']:
            improvement = ((lm_agg['MAP'] - vsm_agg['MAP']) / vsm_agg['MAP']) * 100
            f.write(f"  Unigram Language Model\n")
            f.write(f"  MAP improvement: {improvement:.2f}%\n")
        elif vsm_agg['MAP'] > lm_agg['MAP']:
            improvement = ((vsm_agg['MAP'] - lm_agg['MAP']) / lm_agg['MAP']) * 100
            f.write(f"  Vector Space Model (VSM)\n")
            f.write(f"  MAP improvement: {improvement:.2f}%\n")
        else:
            f.write(f"  Tie - Both models perform equally\n")
        
        f.write("\n")
    
    print(f"✓ Results saved to {filename}")


def save_comprehensive_metrics(vsm_eval, lm_eval, filename="results/metrics.txt"):
    """Save comprehensive metrics with all evaluation measures."""
    with open(filename, 'w') as f:
        vsm_agg = vsm_eval['aggregated']
        lm_agg = lm_eval['aggregated']
        
        f.write("=" * 70 + "\n")
        f.write("COMPREHENSIVE EVALUATION METRICS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total Queries Evaluated: {vsm_eval['num_queries']}\n\n")
        
        # VSM Detailed Results
        f.write("=" * 70 + "\n")
        f.write("VECTOR SPACE MODEL (VSM)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Core Metrics:\n")
        f.write(f"  MAP (Mean Average Precision):  {vsm_agg['MAP']:.4f}\n")
        f.write(f"  R-Precision:                   {vsm_agg['R-Precision']:.4f}\n\n")
        
        for k in [5, 10]:
            f.write(f"Metrics @ K={k}:\n")
            f.write(f"  Precision@{k}:  {vsm_agg[f'P@{k}']:.4f}\n")
            f.write(f"  Recall@{k}:     {vsm_agg[f'R@{k}']:.4f}\n")
            f.write(f"  F1-Score@{k}:   {vsm_agg[f'F1@{k}']:.4f}\n")
            f.write(f"  nDCG@{k}:       {vsm_agg[f'nDCG@{k}']:.4f}\n")
            f.write(f"  ERR@{k}:        {vsm_agg[f'ERR@{k}']:.4f}\n\n")
        
        # Unigram LM Detailed Results
        f.write("=" * 70 + "\n")
        f.write("UNIGRAM LANGUAGE MODEL\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Core Metrics:\n")
        f.write(f"  MAP (Mean Average Precision):  {lm_agg['MAP']:.4f}\n")
        f.write(f"  R-Precision:                   {lm_agg['R-Precision']:.4f}\n\n")
        
        for k in [5, 10]:
            f.write(f"Metrics @ K={k}:\n")
            f.write(f"  Precision@{k}:  {lm_agg[f'P@{k}']:.4f}\n")
            f.write(f"  Recall@{k}:     {lm_agg[f'R@{k}']:.4f}\n")
            f.write(f"  F1-Score@{k}:   {lm_agg[f'F1@{k}']:.4f}\n")
            f.write(f"  nDCG@{k}:       {lm_agg[f'nDCG@{k}']:.4f}\n")
            f.write(f"  ERR@{k}:        {lm_agg[f'ERR@{k}']:.4f}\n\n")
        
        # Detailed Comparison
        f.write("=" * 70 + "\n")
        f.write("DETAILED PERFORMANCE COMPARISON\n")
        f.write("=" * 70 + "\n\n")
        
        # Core metrics comparison
        f.write("Core Metrics Comparison:\n")
        f.write("-" * 70 + "\n")
        
        core_metrics = [
            ('MAP', 'MAP'),
            ('R-Precision', 'R-Precision')
        ]
        
        for display_name, metric_key in core_metrics:
            vsm_val = vsm_agg[metric_key]
            lm_val = lm_agg[metric_key]
            
            f.write(f"\n{display_name}:\n")
            f.write(f"  VSM:        {vsm_val:.4f}\n")
            f.write(f"  Unigram LM: {lm_val:.4f}\n")
            
            if lm_val > vsm_val:
                diff = lm_val - vsm_val
                pct = (diff / vsm_val * 100) if vsm_val > 0 else 0
                f.write(f"  Winner:     Unigram LM (+{diff:.4f}, +{pct:.2f}%)\n")
            elif vsm_val > lm_val:
                diff = vsm_val - lm_val
                pct = (diff / lm_val * 100) if lm_val > 0 else 0
                f.write(f"  Winner:     VSM (+{diff:.4f}, +{pct:.2f}%)\n")
            else:
                f.write(f"  Winner:     Tie\n")
        
        # K-based metrics comparison
        f.write("\n\nK-based Metrics Comparison:\n")
        f.write("-" * 70 + "\n")
        
        for k in [5, 10]:
            f.write(f"\nAt K={k}:\n")
            
            k_metrics = [
                (f'Precision@{k}', f'P@{k}'),
                (f'Recall@{k}', f'R@{k}'),
                (f'F1-Score@{k}', f'F1@{k}'),
                (f'nDCG@{k}', f'nDCG@{k}'),
                (f'ERR@{k}', f'ERR@{k}')
            ]
            
            for display_name, metric_key in k_metrics:
                vsm_val = vsm_agg[metric_key]
                lm_val = lm_agg[metric_key]
                
                f.write(f"  {display_name:15s}: VSM={vsm_val:.4f}, Unigram={lm_val:.4f}")
                
                if abs(lm_val - vsm_val) > 0.0001:  # Significant difference
                    if lm_val > vsm_val:
                        pct = ((lm_val - vsm_val) / vsm_val * 100) if vsm_val > 0 else 0
                        f.write(f"  [Unigram +{pct:.1f}%]")
                    else:
                        pct = ((vsm_val - lm_val) / lm_val * 100) if lm_val > 0 else 0
                        f.write(f"  [VSM +{pct:.1f}%]")
                
                f.write("\n")
        
        # Overall Winner
        f.write("\n" + "=" * 70 + "\n")
        f.write("OVERALL WINNER (Based on MAP)\n")
        f.write("=" * 70 + "\n\n")
        
        if lm_agg['MAP'] > vsm_agg['MAP']:
            improvement = ((lm_agg['MAP'] - vsm_agg['MAP']) / vsm_agg['MAP']) * 100
            f.write(f"Winner: Unigram Language Model\n")
            f.write(f"  MAP Improvement: {improvement:.2f}%\n")
            f.write(f"  Absolute Difference: {lm_agg['MAP'] - vsm_agg['MAP']:.4f}\n")
        elif vsm_agg['MAP'] > lm_agg['MAP']:
            improvement = ((vsm_agg['MAP'] - lm_agg['MAP']) / lm_agg['MAP']) * 100
            f.write(f"Winner: Vector Space Model (VSM)\n")
            f.write(f"  MAP Improvement: {improvement:.2f}%\n")
            f.write(f"  Absolute Difference: {vsm_agg['MAP'] - lm_agg['MAP']:.4f}\n")
        else:
            f.write(f"Result: Tie - Both models perform equally\n")
        
        f.write("\n")
    
    print(f"✓ Comprehensive metrics saved to {filename}")


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
    
    print(f"✓ Detailed rankings saved to {filename}")


def print_results_summary(vsm_eval, lm_eval):
    vsm_agg = vsm_eval['aggregated']
    lm_agg = lm_eval['aggregated']
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print("Quick Comparison:")
    print(f"  VSM:      P@5={vsm_agg['P@5']:.4f}, MAP={vsm_agg['MAP']:.4f}, nDCG@10={vsm_agg['nDCG@10']:.4f}")
    print(f"  Unigram:  P@5={lm_agg['P@5']:.4f}, MAP={lm_agg['MAP']:.4f}, nDCG@10={lm_agg['nDCG@10']:.4f}")
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
    
    print(f"\n✓ Processed {len(queries)} queries with both models")
    
    # ========================================================================
    # STEP 7: Compare Models
    # ========================================================================
    compare_models(vsm_results, lm_results, queries, num_examples=2)
    
    # ========================================================================
    # STEP 8: Comprehensive Evaluation
    # ========================================================================
    print("\n[STEP 8] Running Comprehensive Evaluation...")
    
    vsm_eval = evaluate_model(
        "Vector Space Model (VSM)",
        queries,
        relevances,
        vsm_results,
        k_values=[5, 10]
    )
    
    lm_eval = evaluate_model(
        "Unigram Language Model (Dirichlet)",
        queries,
        relevances,
        lm_results,
        k_values=[5, 10]
    )
    
    # ========================================================================
    # STEP 9: Save Results
    # ========================================================================
    print("\n[STEP 9] Saving Results...")
    
    import os
    os.makedirs("results", exist_ok=True)
    
    # Save original simple results file (for backward compatibility)
    save_unified_results(vsm_eval, lm_eval, "results/results.txt")
    
    # Save comprehensive metrics file (NEW - with all metrics)
    save_comprehensive_metrics(vsm_eval, lm_eval, "results/metrics.txt")
    
    # Save detailed rankings
    save_detailed_rankings(vsm_results, lm_results, queries, "results/detailed_rankings.txt")
    
    # ========================================================================
    # STEP 10: Display Summary
    # ========================================================================
    print_results_summary(vsm_eval, lm_eval)
    
    print("\n" + "=" * 70)
    print("✓ SYSTEM EXECUTION COMPLETE")
    print("=" * 70)
    
    print("\nGenerated Files:")
    print("  results/results.txt           - Simple evaluation summary")
    print("  results/metrics.txt           - Comprehensive metrics & comparison")
    print("  results/detailed_rankings.txt - Detailed query rankings")
    
    print("\nKey Metrics:")
    vsm_agg = vsm_eval['aggregated']
    lm_agg = lm_eval['aggregated']
    
    print(f"\n  VSM:")
    print(f"    MAP: {vsm_agg['MAP']:.4f}  |  P@5: {vsm_agg['P@5']:.4f}  |  P@10: {vsm_agg['P@10']:.4f}")
    print(f"    nDCG@10: {vsm_agg['nDCG@10']:.4f}  |  ERR@10: {vsm_agg['ERR@10']:.4f}")
    
    print(f"\n  Unigram LM:")
    print(f"    MAP: {lm_agg['MAP']:.4f}  |  P@5: {lm_agg['P@5']:.4f}  |  P@10: {lm_agg['P@10']:.4f}")
    print(f"    nDCG@10: {lm_agg['nDCG@10']:.4f}  |  ERR@10: {lm_agg['ERR@10']:.4f}")
    
    # Determine winner
    print(f"\n  Overall Winner (MAP):")
    if lm_agg['MAP'] > vsm_agg['MAP']:
        improvement = ((lm_agg['MAP'] - vsm_agg['MAP']) / vsm_agg['MAP']) * 100
        print(f"    Unigram Language Model ({improvement:.2f}% better)")
    elif vsm_agg['MAP'] > lm_agg['MAP']:
        improvement = ((vsm_agg['MAP'] - lm_agg['MAP']) / lm_agg['MAP']) * 100
        print(f"    VSM ({improvement:.2f}% better)")
    else:
        print(f"    Tie: Both models perform equally")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure Cranfield dataset is in data/cranfield/")
        print("Required files:")
        print("  - cran.all.1400")
        print("  - cran.qry")
        print("  - cranqrel")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()