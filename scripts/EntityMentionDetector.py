import re
from collections import defaultdict
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
import multiprocessing as mp
from tqdm import tqdm
import pickle
import os

# Global instance for multiprocessing
global_detector = None

def init_worker(detector):
    global global_detector
    global_detector = detector

def worker_process_conversation(batch):
    results = []
    for conv in batch:
        conv_results = []
        for turn in conv:
            mentions = global_detector.detect_mentions(turn['content'])
            conv_results.append({
                'role': turn['role'],
                'content': turn['content'],
                'mentions': sorted(mentions.items(), key=lambda x: x[1], reverse=True)
            })
        results.append(conv_results)
    return results


class EntityMentionDetector:
    def __init__(self, entities: List[str], ngram_size: int = 2, threshold: float = 0.7):
        self.entities = [e.lower() for e in entities]
        self.ngram_size = ngram_size
        self.threshold = threshold

        self.entity_ngrams = {}
        self.entity_words = {}
        self.entity_regex = {}
        self._precompute_entity_data()

    def _precompute_entity_data(self):
        for entity in self.entities:
            self.entity_regex[entity] = re.compile(r'\b' + re.escape(entity) + r'\b', re.IGNORECASE)
            self.entity_words[entity] = set(entity.lower().split())
            self.entity_ngrams[entity] = self._get_ngrams(entity)

    def _get_ngrams(self, text: str) -> Set[str]:
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = ' '.join(text.split())  # Normalize whitespace
        return {text[i:i + self.ngram_size] for i in range(len(text) - self.ngram_size + 1)}

    def _quick_word_filter(self, text_words: Set[str], entity: str) -> bool:
        entity_words = self.entity_words[entity]
        return not entity_words.isdisjoint(text_words)

    def detect_mentions(self, text: str) -> Dict[str, float]:
        if not text or not text.strip():
            return {}

        text_lower = text.lower()
        text_words = set(text_lower.split())
        results = {}
        text_ngrams = None  # Lazy init

        for entity in self.entities:
            if not self._quick_word_filter(text_words, entity):
                continue

            if self.entity_regex[entity].search(text_lower):
                results[entity] = 1.0
                continue

            if text_ngrams is None:
                text_ngrams = self._get_ngrams(text_lower)

            entity_ngrams = self.entity_ngrams[entity]
            if len(entity_ngrams) < 2:
                continue

            intersection = len(text_ngrams & entity_ngrams)
            if intersection < self.threshold * len(entity_ngrams):
                continue

            union = len(text_ngrams | entity_ngrams)
            similarity = intersection / union if union else 0.0

            if similarity >= self.threshold:
                results[entity] = similarity

        return results

    def process_conversations(self, conversations: List[List[Dict[str, str]]], batch_size: int = 1000, 
                              num_processes: Optional[int] = None, show_progress: bool = True,
                              cache_file: Optional[str] = None) -> List[List[Dict]]:
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass

        if num_processes is None:
            num_processes = max(1, mp.cpu_count() - 1)

        batches = [conversations[i:i + batch_size] for i in range(0, len(conversations), batch_size)]
        results = []

        if num_processes <= 1:
            for batch in tqdm(batches, desc="Processing conversations", disable=not show_progress):
                results.extend(worker_process_conversation(batch))
        else:
            with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(self,)) as pool:
                func = pool.imap if show_progress else pool.map
                iterable = func(worker_process_conversation, batches)
                for batch_result in tqdm(iterable, total=len(batches), disable=not show_progress,
                                         desc=f"Processing conversation batches ({num_processes} workers)"):
                    results.extend(batch_result)

        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)

        return results
