"""
Custom Evaluator for User-Provided Question-Answer Pairs
Allows users to input their own Q&A pairs and visualize metrics
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import numpy as np

from .metrics import RAGEvaluator, EvaluationResult


class CustomQAEvaluator:
    """Evaluator for custom question-answer pairs"""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the custom evaluator
        
        Args:
            embedding_model_name: Name of the sentence transformer model to use
        """
        self.evaluator = RAGEvaluator(embedding_model_name)
        self._ensure_rag_initialized()
    
    def _ensure_rag_initialized(self):
        """Ensure RAG components are initialized"""
        try:
            from rag import globals as rag_globals
            
            # Check if RAG components are already loaded
            if rag_globals.embeddings is None or rag_globals.index is None or rag_globals.documents is None:
                print("🔧 RAG components not initialized, loading them now...")
                
                # Load embedding model
                if rag_globals.embeddings is None:
                    print("🤖 Loading embedding model...")
                    from rag.embedding import load_embedding_model
                    rag_globals.embeddings = load_embedding_model()
                
                # Load FAISS index
                if rag_globals.index is None:
                    print("📦 Loading FAISS index...")
                    from rag.vector_store import load_faiss_index
                    rag_globals.index = load_faiss_index()
                
                # Load documents
                if rag_globals.documents is None:
                    print("📚 Loading documents...")
                    from rag.content_retriever import load_documents
                    rag_globals.documents = load_documents()
                
                print("✅ RAG components initialized successfully")
            else:
                print("✅ RAG components already initialized")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize RAG components: {e}")
            print("   Evaluation will proceed but RAG-generated answers may not be available")
        
    def evaluate_custom_qa_pairs(self, qa_pairs: List[Dict], 
                                get_context_from_rag: bool = True,
                                output_dir: str = "evaluation_results") -> Dict:
        """
        Evaluate custom question-answer pairs
        
        Args:
            qa_pairs: List of dictionaries with 'question' and 'answer' keys
            get_context_from_rag: Whether to get context from RAG system or use provided context
            output_dir: Directory to save results
            
        Returns:
            Dictionary with evaluation results
        """
        print("🔍 Evaluating custom Q&A pairs...")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        evaluation_results = []
        
        qa_pairs = qa_pairs[:30]
        for i, qa_pair in enumerate(qa_pairs):
            print(f"Processing pair {i+1}/{len(qa_pairs)}: {qa_pair['question'][:50]}...")
            
            question = qa_pair['question']
            ground_truth = qa_pair['answer']  # Use provided answer as ground truth
            
            # Generate answer using RAG system
            try:
                from service.ask_service import ask_llm_with_rag
                generated_answer, context = ask_llm_with_rag(question)
                print(f"  Generated answer: {generated_answer[:100]}...")
            except Exception as e:
                print(f"Warning: Could not generate answer via RAG for pair {i+1}: {e}")
                return
                # Fallback: use provided answer and try to get context
                # generated_answer = ground_truth
                # if get_context_from_rag:
                #     try:
                #         from rag.embedding import embed_query
                #         from rag.vector_store import search_similar_chunks
                #         from rag import globals as rag_globals
                        
                #         query_vec = embed_query(question, rag_globals.embeddings)
                #         top_chunks = search_similar_chunks(rag_globals.index, query_vec, rag_globals.documents, k=3)
                #         context = "\n\n".join(top_chunks)
                #     except Exception as e2:
                #         print(f"Fallback Warning: Could not get context from RAG for pair {i+1}: {e2}")
                #         context = qa_pair.get('context', '')
                # else:
                #     context = qa_pair.get('context', '')
            
            # Evaluate the pair
            try:
                result = self.evaluator.evaluate_single_response(
                    question=question,
                    answer=generated_answer,
                    context=context,
                    ground_truth=ground_truth
                )
                
                # Add metadata
                result.pair_id = i + 1
                
                evaluation_results.append(result)
                
            except Exception as e:
                print(f"Error evaluating pair {i+1}: {e}")
                continue
        
        # Calculate aggregate metrics
        aggregate_metrics = self.evaluator.calculate_aggregate_metrics(evaluation_results)
        
        # Generate reports and visualizations
        reports = self._generate_custom_reports(evaluation_results, aggregate_metrics, output_dir)
        self._create_custom_visualizations(evaluation_results, aggregate_metrics, output_dir)
        
        return {
            'evaluation_results': [r.to_dict() for r in evaluation_results],
            'aggregate_metrics': aggregate_metrics,
            'reports': reports,
            'total_pairs': len(qa_pairs),
            'successful_evaluations': len(evaluation_results)
        }
    
    def _generate_custom_reports(self, evaluation_results: List[EvaluationResult], 
                               aggregate_metrics: Dict[str, float], 
                               output_dir: str) -> Dict:
        """Generate reports for custom evaluation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Summary report
        summary_report = {
            'timestamp': datetime.now().isoformat(),
            'total_evaluations': len(evaluation_results),
            'overall_scores': {
                'groundedness': aggregate_metrics.get('avg_groundedness', 0),
                'relevance': aggregate_metrics.get('avg_relevance', 0),
                'factual_consistency': aggregate_metrics.get('avg_factual_consistency', 0),
                'hallucination_detection': aggregate_metrics.get('avg_hallucination_detection', 0),
                'context_utilization': aggregate_metrics.get('avg_context_utilization', 0),
                'truth_similarity': aggregate_metrics.get('avg_truth_similarity', 0)
            }
        }
        
        # Detailed results
        detailed_results = []
        for result in evaluation_results:
            detailed_results.append({
                'pair_id': getattr(result, 'pair_id', 0),
                'question': result.question,
                'answer': result.generated_answer,
                'context': result.retrieved_context,
                'metrics': {
                    'groundedness_score': result.groundedness_score,
                    'relevance_score': result.relevance_score,
                    'factual_consistency': result.factual_consistency,
                    'hallucination_detection': result.hallucination_detection,
                    'context_utilization': result.context_utilization,
                    'answer_length': result.answer_length,
                    'truth_similarity': result.truth_similarity
                },

            })
        
        detailed_report = {
            'timestamp': datetime.now().isoformat(),
            'total_results': len(detailed_results),
            'detailed_results': detailed_results
        }
        
        # Save reports
        with open(f"{output_dir}/custom_summary_report_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=2)
        
        with open(f"{output_dir}/custom_detailed_report_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, ensure_ascii=False, indent=2)
        
        return {
            'summary': summary_report,
            'detailed': detailed_report
        }
    
    def _create_custom_visualizations(self, evaluation_results: List[EvaluationResult], 
                                    aggregate_metrics: Dict[str, float], 
                                    output_dir: str):
        """Create visualizations for custom evaluation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Individual Pair Performance
        self._create_pair_performance_chart(evaluation_results, output_dir, timestamp)
        
        # 2. Metrics Distribution
        self._create_metrics_distribution_chart(evaluation_results, output_dir, timestamp)
        
        # 3. Groundedness vs Relevance Scatter
        self._create_custom_scatter_plot(evaluation_results, output_dir, timestamp)
        
        # 4. Overall Metrics Summary
        self._create_custom_overall_metrics_chart(aggregate_metrics, output_dir, timestamp)
        

    
    def _create_pair_performance_chart(self, evaluation_results: List[EvaluationResult], 
                                     output_dir: str, timestamp: str):
        """Create chart showing performance for each pair"""
        pair_ids = [getattr(r, 'pair_id', i+1) for i, r in enumerate(evaluation_results)]
        groundedness_scores = [r.groundedness_score for r in evaluation_results]
        relevance_scores = [r.relevance_score for r in evaluation_results]
        
        x = np.arange(len(pair_ids))
        width = 0.35
        
        plt.figure(figsize=(max(12, len(pair_ids) * 0.8), 6))
        plt.bar(x - width/2, groundedness_scores, width, label='Groundedness', color='#2E86AB', alpha=0.8)
        plt.bar(x + width/2, relevance_scores, width, label='Relevance', color='#A23B72', alpha=0.8)
        
        plt.xlabel('Question-Answer Pair', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Performance by Question-Answer Pair', fontsize=16, fontweight='bold')
        plt.xticks(x, [f'Pair {pid}' for pid in pair_ids], rotation=45)
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pair_performance_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_metrics_distribution_chart(self, evaluation_results: List[EvaluationResult], 
                                         output_dir: str, timestamp: str):
        """Create distribution charts for each metric"""
        metrics = ['groundedness_score', 'relevance_score', 'factual_consistency', 
                  'hallucination_detection', 'context_utilization', 'truth_similarity']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            values = [getattr(r, metric, 0) for r in evaluation_results]
            
            axes[i].hist(values, bins=10, alpha=0.7, color='#2E86AB', edgecolor='black')
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            axes[i].set_xlabel('Score')
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(np.mean(values), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(values):.3f}')
            axes[i].legend()
        
        # All subplots are used now with 6 metrics in 2x3 grid
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_distribution_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_custom_scatter_plot(self, evaluation_results: List[EvaluationResult], 
                                  output_dir: str, timestamp: str):
        """Create scatter plot of groundedness vs relevance"""
        groundedness_scores = [r.groundedness_score for r in evaluation_results]
        relevance_scores = [r.relevance_score for r in evaluation_results]
        pair_ids = [getattr(r, 'pair_id', i+1) for i, r in enumerate(evaluation_results)]
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(groundedness_scores, relevance_scores, 
                            c=pair_ids, cmap='viridis', s=100, alpha=0.7)
        plt.xlabel('Groundedness Score', fontsize=12)
        plt.ylabel('Relevance Score', fontsize=12)
        plt.title('Groundedness vs Relevance for Custom Q&A Pairs', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(groundedness_scores, relevance_scores, 1)
        p = np.poly1d(z)
        plt.plot(groundedness_scores, p(groundedness_scores), "r--", alpha=0.8)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Pair ID')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/custom_groundedness_vs_relevance_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_custom_overall_metrics_chart(self, aggregate_metrics: Dict[str, float], 
                                           output_dir: str, timestamp: str):
        """Create overall metrics bar chart"""
        metrics = ['avg_groundedness', 'avg_relevance', 'avg_factual_consistency', 
                  'avg_hallucination_detection', 'avg_context_utilization', 'avg_truth_similarity']
        
        values = [aggregate_metrics.get(metric, 0) for metric in metrics]
        labels = ['Groundedness', 'Relevance', 'Factual Consistency', 
                 'Hallucination Detection', 'Context Utilization', 'Truth Similarity']
        
        plt.figure(figsize=(14, 6))
        bars = plt.bar(labels, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B5B95', '#FF6B35'])
        plt.title('Overall Metrics for Custom Q&A Pairs', fontsize=16, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/custom_overall_metrics_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    



def load_qa_pairs_from_file(filename: str) -> List[Dict]:
    """
    Load question-answer pairs from JSON file
    
    Expected format:
    [
        {
            "question": "What is NLP?",
            "answer": "Natural Language Processing is a field of AI.",
            "context": "Optional context..."
        }
    ]
    """
    with open(filename, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    return qa_pairs


def save_qa_pairs_to_file(qa_pairs: List[Dict], filename: str):
    """Save question-answer pairs to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    print(f"Q&A pairs saved to {filename}")


def create_sample_qa_pairs() -> List[Dict]:
    """Create sample question-answer pairs for testing"""
    return [
        {
            "question": "Các phương pháp xử lý ngôn ngữ tự nhiên chính là gì?",
            "answer": "Các phương pháp xử lý ngôn ngữ tự nhiên chính bao gồm tokenization, stemming, lemmatization và các mô hình deep learning như BERT, GPT."
        },
        {
            "question": "Mô hình transformer hoạt động như thế nào?",
            "answer": "Mô hình transformer sử dụng cơ chế attention để xử lý chuỗi dữ liệu, cho phép mô hình tập trung vào các phần khác nhau của input."
        },
        {
            "question": "So sánh giữa RNN và LSTM trong xử lý chuỗi",
            "answer": "RNN có vấn đề vanishing gradient, trong khi LSTM có cơ chế gate để giải quyết vấn đề này và xử lý chuỗi dài tốt hơn."
        },
        {
            "question": "Tại sao attention mechanism quan trọng trong transformer?",
            "answer": "Attention mechanism cho phép mô hình tập trung vào các phần quan trọng của input, giúp cải thiện hiệu suất xử lý ngôn ngữ tự nhiên."
        },
        {
            "question": "Cách thức hoạt động của hệ thống RAG?",
            "answer": "Hệ thống RAG kết hợp retrieval (tìm kiếm thông tin) và generation (sinh câu trả lời) để tạo ra câu trả lời dựa trên thông tin có sẵn."
        }
    ]


def run_custom_evaluation_example():
    """Run example custom evaluation"""
    print("🚀 Running Custom Q&A Evaluation Example...")
    
    # Create sample Q&A pairs
    qa_pairs = create_sample_qa_pairs()
    save_qa_pairs_to_file(qa_pairs, "sample_qa_pairs.json")
    
    # Run evaluation
    evaluator = CustomQAEvaluator()
    results = evaluator.evaluate_custom_qa_pairs(qa_pairs, get_context_from_rag=True)
    
    print(f"\n📊 Custom Evaluation Results:")
    print(f"Total pairs: {results['total_pairs']}")
    print(f"Successful evaluations: {results['successful_evaluations']}")
    print(f"Average groundedness: {results['aggregate_metrics']['avg_groundedness']:.3f}")
    print(f"Average relevance: {results['aggregate_metrics']['avg_relevance']:.3f}")
    
    return results


if __name__ == "__main__":
    run_custom_evaluation_example()
