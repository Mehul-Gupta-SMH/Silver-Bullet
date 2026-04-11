"""Feature cache backed by SQLite via CacheDB.

Replaces the old per-pair JSON file approach (cache/{md5}.json).
The public interface (get_features / save_features) is unchanged so
callers (train.py, precompute_features.py, predict.py) need no edits.
"""

from typing import Any, Dict

from backend.cache_db import CacheDB


class FeatureCache:
    def __init__(self, cache_dir: str = "cache"):
        # cache_dir kept for API compat but ignored — CacheDB owns the path.
        self._db = CacheDB.get()

    def _generate_key(self, text1: str, text2: str) -> str:
        return CacheDB._md5(text1, text2)

    def get_features(self, text1: str, text2: str) -> Dict[str, Any] | None:
        return self._db.get_features(text1, text2)

    def save_features(self, text1: str, text2: str, features: Dict[str, Any]) -> None:
        self._db.save_features(text1, text2, features)
