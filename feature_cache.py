import json
import os
import hashlib
from typing import Dict, Any

class FeatureCache:
    def __init__(self, cache_dir='cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _generate_key(self, text1: str, text2: str) -> str:
        """Generate a unique key for a pair of texts"""
        combined = f"{text1}|||{text2}"
        return hashlib.md5(combined.encode()).hexdigest()

    def get_features(self, text1: str, text2: str) -> Dict[str, Any]:
        """Get features from cache if they exist"""
        key = self._generate_key(text1, text2)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")

        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def save_features(self, text1: str, text2: str, features: Dict[str, Any]):
        """Save features to cache"""
        key = self._generate_key(text1, text2)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(features, f, ensure_ascii=False, indent=2)
