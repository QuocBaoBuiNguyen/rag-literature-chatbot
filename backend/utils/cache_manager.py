"""
Cache Manager for RAG Literature Chatbot
Provides in-memory and persistent caching for faster responses
"""

import hashlib
import json
import time
import os
import pickle
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from collections import OrderedDict

class CacheManager:
    """Manages caching for RAG responses and embeddings"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize cache manager
        
        Args:
            max_size: Maximum number of cached items
            ttl: Time to live in seconds (1 hour default)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        # Cache file paths
        self.cache_dir = "data/cache"
        self.response_cache_file = os.path.join(self.cache_dir, "response_cache.pkl")
        self.embedding_cache_file = os.path.join(self.cache_dir, "embedding_cache.pkl")
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load existing cache
        self._load_cache()
    
    def _generate_cache_key(self, question: str) -> str:
        """Generate a unique cache key for a question"""
        # Normalize the question for consistent hashing
        normalized_question = question.strip().lower()
        return hashlib.md5(normalized_question.encode('utf-8')).hexdigest()
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cached item is still valid based on TTL"""
        return time.time() - timestamp < self.ttl
    
    def get(self, question: str) -> Optional[str]:
        """
        Get cached response for a question
        
        Args:
            question: The question to look up
            
        Returns:
            Cached response if found and valid, None otherwise
        """
        cache_key = self._generate_cache_key(question)
        self.cache_stats['total_requests'] += 1
        
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            
            # Check if cache is still valid
            if self._is_cache_valid(cached_data['timestamp']):
                # Move to end (most recently used)
                self.cache.move_to_end(cache_key)
                self.cache_stats['hits'] += 1
                print(f"🚀 Cache hit for question: {question[:50]}...")
                return cached_data['response']
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
                print(f"⏰ Cache expired for question: {question[:50]}...")
        
        self.cache_stats['misses'] += 1
        return None
    
    def set(self, question: str, response: str) -> None:
        """
        Cache a response for a question
        
        Args:
            question: The question
            response: The response to cache
        """
        cache_key = self._generate_cache_key(question)
        
        # Remove oldest entry if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.cache_stats['evictions'] += 1
            print(f"🗑️ Cache eviction for oldest entry")
        
        # Add new cache entry
        self.cache[cache_key] = {
            'response': response,
            'timestamp': time.time(),
            'question': question[:100]  # Store first 100 chars for debugging
        }
        
        # Move to end (most recently used)
        self.cache.move_to_end(cache_key)
        print(f"💾 Cached response for question: {question[:50]}...")
    
    def _load_cache(self) -> None:
        """Load cache from disk"""
        try:
            if os.path.exists(self.response_cache_file):
                with open(self.response_cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                    # Load cache entries and filter expired ones
                    for key, data in cached_data.items():
                        if self._is_cache_valid(data['timestamp']):
                            self.cache[key] = data
                    
                    print(f"✅ Loaded {len(self.cache)} valid cache entries")
            else:
                print("ℹ️ No existing cache file found")
        except Exception as e:
            print(f"⚠️ Failed to load cache: {e}")
    
    def _save_cache(self) -> None:
        """Save cache to disk"""
        try:
            with open(self.response_cache_file, 'wb') as f:
                pickle.dump(dict(self.cache), f)
            print(f"✅ Saved {len(self.cache)} cache entries to disk")
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = (self.cache_stats['hits'] / 
                   max(self.cache_stats['total_requests'], 1))
        
        return {
            'total_requests': self.cache_stats['total_requests'],
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'evictions': self.cache_stats['evictions'],
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'ttl': self.ttl
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        self.cache.clear()
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        print("🗑️ Cache cleared")
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries and return count of removed items"""
        expired_count = 0
        expired_keys = []
        
        for key, data in self.cache.items():
            if not self._is_cache_valid(data['timestamp']):
                expired_keys.append(key)
                expired_count += 1
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_count > 0:
            print(f"🧹 Cleaned up {expired_count} expired cache entries")
        
        return expired_count
    
    def shutdown(self) -> None:
        """Save cache before shutdown"""
        self._save_cache()
        print("💾 Cache saved on shutdown")

# Global cache manager instance
cache_manager = CacheManager()

def get_cached_response(question: str) -> Optional[str]:
    """Get cached response for a question"""
    return cache_manager.get(question)

def cache_response(question: str, response: str) -> None:
    """Cache a response for a question"""
    cache_manager.set(question, response)

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    return cache_manager.get_stats()

def clear_cache() -> None:
    """Clear all cached data"""
    cache_manager.clear_cache()

def cleanup_expired_cache() -> int:
    """Clean up expired cache entries"""
    return cache_manager.cleanup_expired() 