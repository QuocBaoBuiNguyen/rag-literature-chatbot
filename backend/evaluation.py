#!/usr/bin/env python3
"""
Custom Q&A Evaluation Runner
Allows users to input their own question-answer pairs and visualize metrics
"""

import sys
import os
import json
from pathlib import Path

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.custom_evaluator import (
    CustomQAEvaluator, 
    load_qa_pairs_from_file, 
    save_qa_pairs_to_file, 
    create_sample_qa_pairs
)


def interactive_qa_input():
    """Interactive input for Q&A pairs"""
    print("📝 Interactive Q&A Pair Input")
    print("=" * 40)
    print("Enter your question-answer pairs. Type 'done' when finished.")
    print("Format: question|answer")
    print("Example: What is NLP?|NLP is a field of AI")
    print()
    
    qa_pairs = []
    pair_num = 1
    
    while True:
        try:
            user_input = input(f"Pair {pair_num} (or 'done' to finish): ").strip()
            
            if user_input.lower() == 'done':
                break
            
            if '|' in user_input:
                parts = user_input.split('|')
                if len(parts) >= 2:
                    question = parts[0].strip()
                    answer = parts[1].strip()
                    
                    qa_pairs.append({
                        'question': question,
                        'answer': answer
                    })
                    pair_num += 1
                else:
                    print("❌ Invalid format. Please use: question|answer")
            else:
                print("❌ Invalid format. Please use: question|answer")
                
        except KeyboardInterrupt:
            print("\n⚠️ Input interrupted.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return qa_pairs


def load_from_file():
    """Load Q&A pairs from file"""
    print("📁 Load Q&A pairs from file")
    print("=" * 30)
    
    filename = input("Enter filename (or press Enter for 'qa_pairs.json'): ").strip()
    if not filename:
        filename = "backend/example_qa_pairs.json"
    
    if not os.path.exists(filename):
        print(f"❌ File '{filename}' not found.")
        return None
    
    try:
        qa_pairs = load_qa_pairs_from_file(filename)
        print(f"✅ Loaded {len(qa_pairs)} Q&A pairs from '{filename}'")
        return qa_pairs
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


def create_sample_file():
    """Create a sample Q&A pairs file"""
    print("📝 Creating sample Q&A pairs file...")
    
    qa_pairs = create_sample_qa_pairs()
    filename = "sample_qa_pairs.json"
    save_qa_pairs_to_file(qa_pairs, filename)
    
    print(f"✅ Sample file created: {filename}")
    print("You can edit this file and then load it for evaluation.")
    
    return qa_pairs


def main():
    """Main function"""
    print("🎯 Custom Q&A Evaluation System")
    print("=" * 40)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "sample":
            qa_pairs = create_sample_file()
            if qa_pairs:
                run_evaluation(qa_pairs)
            return
        elif sys.argv[1] == "file" and len(sys.argv) > 2:
            filename = sys.argv[2]
            try:
                qa_pairs = load_qa_pairs_from_file(filename)
                run_evaluation(qa_pairs)
            except Exception as e:
                print(f"❌ Error loading file: {e}")
            return
    
    # Interactive mode
    print("Choose input method:")
    print("1. Create sample file")
    print("2. Load from file")
    
    choice = input("\nEnter your choice (1-2): ").strip()
    
    qa_pairs = None
    
    if choice == "1":
        qa_pairs = create_sample_file()
    elif choice == "2":
        qa_pairs = load_from_file()
    else:
        print("❌ Invalid choice")
        return
    
    if qa_pairs:
        run_evaluation(qa_pairs)
    else:
        print("❌ No Q&A pairs to evaluate")


def run_evaluation(qa_pairs):
    """Run evaluation on Q&A pairs"""
    print(f"\n🚀 Running evaluation on {len(qa_pairs)} Q&A pairs...")
    
    try:
        # Ask user for evaluation options
        print("\nEvaluation options:")
        get_context_from_rag = input("Get context from RAG system? (y/n, default: y): ").strip().lower() != 'n'
        
        output_dir = input("Output directory (default: evaluation_results): ").strip()
        if not output_dir:
            output_dir = "evaluation_results"
        
        # Run evaluation
        evaluator = CustomQAEvaluator()
        results = evaluator.evaluate_custom_qa_pairs(
            qa_pairs=qa_pairs,
            get_context_from_rag=get_context_from_rag,
            output_dir=output_dir
        )
        
        # Display results
        print(f"\n📊 Evaluation Results:")
        print("=" * 30)
        print(f"Total pairs: {results['total_pairs']}")
        print(f"Successful evaluations: {results['successful_evaluations']}")
        print(f"Average groundedness: {results['aggregate_metrics']['avg_groundedness']:.3f}")
        print(f"Average relevance: {results['aggregate_metrics']['avg_relevance']:.3f}")
        print(f"Average factual consistency: {results['aggregate_metrics']['avg_factual_consistency']:.3f}")
        print(f"Average hallucination detection: {results['aggregate_metrics']['avg_hallucination_detection']:.3f}")
        print(f"Average context utilization: {results['aggregate_metrics']['avg_context_utilization']:.3f}")
        print(f"Average truth similarity: {results['aggregate_metrics']['avg_truth_similarity']:.3f}")
        
        print(f"\n📁 Results saved to: {output_dir}/")
        print("📈 Visualizations created:")
        print("  - pair_performance_*.png")
        print("  - metrics_distribution_*.png")
        print("  - custom_groundedness_vs_relevance_*.png")
        print("  - custom_overall_metrics_*.png")
        
        # Show individual results
        print(f"\n📋 Individual Results:")
        print("-" * 50)
        for i, result in enumerate(results['evaluation_results']):
            print(f"Pair {i+1}:")
            print(f"  Question: {result['question'][:50]}...")
            print(f"  Groundedness: {result['groundedness_score']:.3f}")
            print(f"  Relevance: {result['relevance_score']:.3f}")
            print(f"  Truth Similarity: {result['truth_similarity']:.3f}")
            print()
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None


def show_usage():
    """Show usage information"""
    print("Usage:")
    print("  python3 run_custom_evaluation.py                    # Interactive mode")
    print("  python3 run_custom_evaluation.py sample            # Create and use sample data")
    print("  python3 run_custom_evaluation.py file <filename>   # Load from file")
    print()
    print("File format (JSON):")
    print("  [")
    print("    {")
    print('      "question": "What is NLP?",')
    print('      "answer": "NLP is a field of AI."')
    print("    }")
    print("  ]")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        show_usage()
    else:
        main()
