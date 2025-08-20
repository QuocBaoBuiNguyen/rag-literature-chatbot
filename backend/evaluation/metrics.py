"""
RAG Evaluation Metrics
Implements groundedness and relevance metrics for RAG system evaluation
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from llm.chatbot_llm import generate_answer

load_dotenv()

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    question: str
    retrieved_context: str
    generated_answer: str
    ground_truth: Optional[str] = None
    
    # Groundedness metrics
    groundedness_score: float = 0.0
    factual_consistency: float = 0.0
    hallucination_detection: float = 0.0
    
    # Relevance metrics
    relevance_score: float = 0.0
    context_relevance: float = 0.0
    answer_relevance: float = 0.0
    
    # Additional metrics
    answer_length: int = 0
    context_utilization: float = 0.0
    truth_similarity: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'question': self.question,
            'retrieved_context': self.retrieved_context,
            'generated_answer': self.generated_answer,
            'ground_truth': self.ground_truth,
            'groundedness_score': self.groundedness_score,
            'factual_consistency': self.factual_consistency,
            'hallucination_detection': self.hallucination_detection,
            'relevance_score': self.relevance_score,
            'context_relevance': self.context_relevance,
            'answer_relevance': self.answer_relevance,
            'answer_length': self.answer_length,
            'context_utilization': self.context_utilization,
            'truth_similarity': self.truth_similarity
        }


class RAGEvaluator:
    """Main class for RAG evaluation"""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the RAG evaluator
        
        Args:
            embedding_model_name: Name of the sentence transformer model to use
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
    def evaluate_groundedness(self, answer: str, context: str) -> Dict[str, float]:
        """
        Evaluate groundedness of the answer against the retrieved context
        
        Args:
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Dictionary containing groundedness metrics
        """
        # 1. Factual Consistency Check
        factual_consistency = self._check_factual_consistency(answer, context)
        
        # 2. Hallucination Detection
        hallucination_score = self._detect_hallucinations(answer, context)
        
        # 3. Context Utilization
        context_utilization = self._calculate_context_utilization(answer, context)
        
        # 4. Overall Groundedness Score
        groundedness_score = (factual_consistency + hallucination_score + context_utilization) / 3
        
        return {
            'groundedness_score': groundedness_score,
            'factual_consistency': factual_consistency,
            'hallucination_detection': hallucination_score,
            'context_utilization': context_utilization
        }
    
    def evaluate_relevance(self, question: str, answer: str, context: str) -> Dict[str, float]:
        """
        Evaluate relevance of the answer to the question and context
        
        Args:
            question: User question
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Dictionary containing relevance metrics
        """
        # 1. Question-Answer Relevance
        answer_relevance = self._calculate_semantic_similarity(question, answer)
        
        # 2. Context-Question Relevance
        context_relevance = self._calculate_semantic_similarity(question, context)
        
        # 3. Overall Relevance Score
        relevance_score = (answer_relevance + context_relevance) / 2
        
        return {
            'relevance_score': relevance_score,
            'answer_relevance': answer_relevance,
            'context_relevance': context_relevance
        }
    
    def _check_factual_consistency(self, answer: str, context: str) -> float:
        """
        Check if the answer is factually consistent with the context using LLM
        """
        prompt = f"""
        Đánh giá tính nhất quán thực tế giữa câu trả lời và ngữ cảnh được cung cấp.
        
        Ngữ cảnh: {context}
        
        Câu trả lời: {answer}
        
        Hãy đánh giá từ 0 đến 1, trong đó:
        0 = Hoàn toàn không nhất quán, chứa thông tin sai lệch
        1 = Hoàn toàn nhất quán, chỉ chứa thông tin từ ngữ cảnh
        
        Chỉ trả về số từ 0 đến 1, không có giải thích khác.
        """
        
        try:
            response_text = generate_answer(prompt)
            score = float(response_text.strip())
            return max(0.0, min(1.0, score))
        except:
            # Fallback to semantic similarity
            return self._calculate_semantic_similarity(answer, context)
    
    def _detect_hallucinations(self, answer: str, context: str) -> float:
        """
        Detect hallucinations in the answer using LLM
        """
        prompt = f"""
        Phát hiện thông tin không có trong ngữ cảnh (hallucination) trong câu trả lời.
        
        Ngữ cảnh: {context}
        
        Câu trả lời: {answer}
        
        Hãy đánh giá từ 0 đến 1, trong đó:
        0 = Chứa nhiều thông tin không có trong ngữ cảnh
        1 = Chỉ chứa thông tin từ ngữ cảnh, không có hallucination
        
        Chỉ trả về số từ 0 đến 1, không có giải thích khác.
        """
        
        try:
            response_text = generate_answer(prompt)
            score = float(response_text.strip())
            return max(0.0, min(1.0, score))
        except:
            # Fallback to keyword overlap
            return self._calculate_keyword_overlap(answer, context)
    
    def _calculate_context_utilization(self, answer: str, context: str) -> float:
        """
        Calculate how much of the context is utilized in the answer
        """
        # Split into sentences
        answer_sentences = re.split(r'[.!?]+', answer.strip())
        context_sentences = re.split(r'[.!?]+', context.strip())
        
        if not answer_sentences or not context_sentences:
            return 0.0
        
        # Calculate semantic similarity between answer and context sentences
        answer_embeddings = self.embedding_model.encode(answer_sentences)
        context_embeddings = self.embedding_model.encode(context_sentences)
        
        similarities = cosine_similarity(answer_embeddings, context_embeddings)
        
        # Average similarity
        avg_similarity = np.mean(similarities)
        return float(avg_similarity)
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using embeddings
        """
        embeddings1 = self.embedding_model.encode([text1])
        embeddings2 = self.embedding_model.encode([text2])
        
        similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
        return float(similarity)
    
    def _calculate_keyword_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate keyword overlap between two texts
        """
        # Extract keywords (simple approach)
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def evaluate_single_response(self, question: str, answer: str, context: str, 
                               ground_truth: Optional[str] = None) -> EvaluationResult:
        """
        Evaluate a single RAG response
        
        Args:
            question: User question
            answer: Generated answer
            context: Retrieved context
            ground_truth: Optional ground truth answer
            
        Returns:
            EvaluationResult object
        """
        # Evaluate groundedness
        groundedness_metrics = self.evaluate_groundedness(answer, context)
        
        # Evaluate relevance
        relevance_metrics = self.evaluate_relevance(question, answer, context)
        
        # Calculate additional metrics
        answer_length = len(answer.split())
        context_utilization = groundedness_metrics['context_utilization']
        
        # Calculate truth similarity if ground truth is available
        truth_similarity = 0.0
        if ground_truth:
            truth_similarity = self._calculate_semantic_similarity(answer, ground_truth)
        
        return EvaluationResult(
            question=question,
            retrieved_context=context,
            generated_answer=answer,
            ground_truth=ground_truth,
            groundedness_score=groundedness_metrics['groundedness_score'],
            factual_consistency=groundedness_metrics['factual_consistency'],
            hallucination_detection=groundedness_metrics['hallucination_detection'],
            relevance_score=relevance_metrics['relevance_score'],
            context_relevance=relevance_metrics['context_relevance'],
            answer_relevance=relevance_metrics['answer_relevance'],
            answer_length=answer_length,
            context_utilization=context_utilization,
            truth_similarity=truth_similarity
        )
    
    def evaluate_batch(self, test_cases: List[Dict]) -> List[EvaluationResult]:
        """
        Evaluate multiple RAG responses
        
        Args:
            test_cases: List of dictionaries with keys: question, answer, context, ground_truth
            
        Returns:
            List of EvaluationResult objects
        """
        results = []
        
        for test_case in test_cases:
            result = self.evaluate_single_response(
                question=test_case['question'],
                answer=test_case['answer'],
                context=test_case['context'],
                ground_truth=test_case.get('ground_truth')
            )
            results.append(result)
        
        return results
    
    def calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """
        Calculate aggregate metrics from evaluation results
        
        Args:
            results: List of EvaluationResult objects
            
        Returns:
            Dictionary with aggregate metrics
        """
        if not results:
            return {}
        
        metrics = {
            'avg_groundedness': np.mean([r.groundedness_score for r in results]),
            'avg_factual_consistency': np.mean([r.factual_consistency for r in results]),
            'avg_hallucination_detection': np.mean([r.hallucination_detection for r in results]),
            'avg_relevance': np.mean([r.relevance_score for r in results]),
            'avg_context_relevance': np.mean([r.context_relevance for r in results]),
            'avg_answer_relevance': np.mean([r.answer_relevance for r in results]),
            'avg_answer_length': np.mean([r.answer_length for r in results]),
            'avg_context_utilization': np.mean([r.context_utilization for r in results]),
            'avg_truth_similarity': np.mean([r.truth_similarity for r in results]),
            'total_evaluations': len(results)
        }
        
        return metrics
