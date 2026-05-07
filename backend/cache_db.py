"""Unified SQLite-backed cache for all SilverBullet inference results.

Replaces four separate file-based caches:
  - cache/{md5}.json          → features table
  - cache/nli_pairs.json      → nli_pairs table
  - cache/entity_sentences.json → entities table
  - cache/relex_triplets.json → triplets table
  - cache/embeddings/*.npz    → embeddings table

Design
------
- Single DB file: cache/silverbullet.db
- WAL journal mode: concurrent readers never block; single writer at a time.
- Thread-local connections: safe for ThreadPoolExecutor (predict.py).
- Bulk in-memory load on first access: each cache group is loaded into its
  extractor's in-memory dict once at process start, then SQLite is only
  written when new items are computed. This keeps hot-path inference fast.
- Automatic one-time migration: old file caches are imported and renamed
  on the first run.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np

DB_PATH = Path("cache/silverbullet.db")

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS embeddings (
    model     TEXT NOT NULL,
    sentence  TEXT NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (model, sentence)
);

CREATE TABLE IF NOT EXISTS nli_pairs (
    sent1         TEXT NOT NULL,
    sent2         TEXT NOT NULL,
    entailment    REAL NOT NULL,
    neutral       REAL NOT NULL,
    contradiction REAL NOT NULL,
    PRIMARY KEY (sent1, sent2)
);

CREATE TABLE IF NOT EXISTS entities (
    sentence TEXT PRIMARY KEY NOT NULL,
    counts   TEXT NOT NULL,
    strings  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS triplets (
    sentence TEXT PRIMARY KEY NOT NULL,
    data     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS efg_pairs (
    sent1    TEXT NOT NULL,
    sent2    TEXT NOT NULL,
    supports REAL NOT NULL,
    refutes  REAL NOT NULL,
    nei      REAL NOT NULL,
    PRIMARY KEY (sent1, sent2)
);

CREATE TABLE IF NOT EXISTS features (
    md5   TEXT PRIMARY KEY NOT NULL,
    text1 TEXT,
    text2 TEXT,
    data  TEXT NOT NULL
);
"""

_local = threading.local()


def _get_conn(db_path: Path) -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating it on first use."""
    if getattr(_local, "conn", None) is None:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
    return _local.conn


class CacheDB:
    """Unified cache interface backed by a single SQLite database.

    Use the singleton:
        db = CacheDB.get()
    """

    _instance: CacheDB | None = None

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Bootstrap: create tables if they don't exist yet.
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.executescript(_DDL)
        conn.commit()
        conn.close()
        self._migrate_old_caches()

    @classmethod
    def get(cls) -> "CacheDB":
        """Return the process-wide singleton, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _conn(self) -> sqlite3.Connection:
        return _get_conn(self.db_path)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def load_all_embeddings(self, model: str) -> dict[str, np.ndarray]:
        """Bulk-load all stored embeddings for *model* into a dict."""
        rows = self._conn().execute(
            "SELECT sentence, embedding FROM embeddings WHERE model=?", (model,)
        ).fetchall()
        return {r[0]: np.frombuffer(r[1], dtype=np.float32).copy() for r in rows}

    def save_embeddings_batch(
        self, model: str, sentences: list[str], embeddings: list[np.ndarray]
    ) -> None:
        rows = [
            (model, s, e.astype(np.float32).tobytes())
            for s, e in zip(sentences, embeddings)
        ]
        conn = self._conn()
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (model, sentence, embedding) VALUES (?,?,?)",
            rows,
        )
        conn.commit()

    # ------------------------------------------------------------------
    # NLI pairs
    # ------------------------------------------------------------------

    def load_all_nli(self) -> dict[tuple[str, str], tuple[float, float, float]]:
        rows = self._conn().execute(
            "SELECT sent1, sent2, entailment, neutral, contradiction FROM nli_pairs"
        ).fetchall()
        return {(r[0], r[1]): (r[2], r[3], r[4]) for r in rows}

    def save_nli_batch(
        self, rows: list[tuple[str, str, float, float, float]]
    ) -> None:
        """rows: [(sent1, sent2, entailment, neutral, contradiction), ...]"""
        conn = self._conn()
        conn.executemany(
            "INSERT OR REPLACE INTO nli_pairs "
            "(sent1, sent2, entailment, neutral, contradiction) VALUES (?,?,?,?,?)",
            rows,
        )
        conn.commit()

    # ------------------------------------------------------------------
    # EFG (External Factual Grounding) pair cache
    # ------------------------------------------------------------------

    def load_all_efg(self) -> dict[tuple[str, str], tuple[float, float, float]]:
        rows = self._conn().execute(
            "SELECT sent1, sent2, supports, refutes, nei FROM efg_pairs"
        ).fetchall()
        return {(r[0], r[1]): (r[2], r[3], r[4]) for r in rows}

    def save_efg_batch(
        self, rows: list[tuple[str, str, float, float, float]]
    ) -> None:
        """rows: [(sent1, sent2, supports, refutes, nei), ...]"""
        conn = self._conn()
        conn.executemany(
            "INSERT OR REPLACE INTO efg_pairs "
            "(sent1, sent2, supports, refutes, nei) VALUES (?,?,?,?,?)",
            rows,
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Entity cache
    # ------------------------------------------------------------------

    def load_all_entities(self) -> dict[str, tuple[dict, dict]]:
        rows = self._conn().execute(
            "SELECT sentence, counts, strings FROM entities"
        ).fetchall()
        return {r[0]: (json.loads(r[1]), json.loads(r[2])) for r in rows}

    def save_entities_batch(
        self,
        sentences: list[str],
        counts_list: list[dict],
        strings_list: list[dict],
    ) -> None:
        rows = [
            (s, json.dumps(c), json.dumps(st))
            for s, c, st in zip(sentences, counts_list, strings_list)
        ]
        conn = self._conn()
        conn.executemany(
            "INSERT OR REPLACE INTO entities (sentence, counts, strings) VALUES (?,?,?)",
            rows,
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Relex triplets
    # ------------------------------------------------------------------

    def load_all_triplets(self) -> dict[str, list]:
        rows = self._conn().execute(
            "SELECT sentence, data FROM triplets"
        ).fetchall()
        return {r[0]: json.loads(r[1]) for r in rows}

    def save_triplets_batch(
        self, sentences: list[str], triplets_list: list[list]
    ) -> None:
        rows = [(s, json.dumps(t)) for s, t in zip(sentences, triplets_list)]
        conn = self._conn()
        conn.executemany(
            "INSERT OR REPLACE INTO triplets (sentence, data) VALUES (?,?)",
            rows,
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Feature maps  (replaces FeatureCache JSON files)
    # ------------------------------------------------------------------

    @staticmethod
    def _md5(text1: str, text2: str) -> str:
        return hashlib.md5(f"{text1}|||{text2}".encode()).hexdigest()

    def get_features(self, text1: str, text2: str) -> dict | None:
        md5 = self._md5(text1, text2)
        row = self._conn().execute(
            "SELECT data FROM features WHERE md5=?", (md5,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def save_features(self, text1: str, text2: str, data: dict) -> None:
        md5 = self._md5(text1, text2)
        self._conn().execute(
            "INSERT OR REPLACE INTO features (md5, text1, text2, data) VALUES (?,?,?,?)",
            (md5, text1, text2, json.dumps(data, ensure_ascii=False)),
        )
        self._conn().commit()

    # ------------------------------------------------------------------
    # One-time migration from old file-based caches
    # ------------------------------------------------------------------

    def _migrate_old_caches(self) -> None:
        migrated = 0

        # --- NLI pairs JSON ---
        nli_file = Path("cache/nli_pairs.json")
        if nli_file.exists():
            try:
                raw = json.loads(nli_file.read_text(encoding="utf-8"))
                rows = []
                for k, v in raw.items():
                    s1, s2 = json.loads(k)
                    rows.append((s1, s2, float(v[0]), float(v[1]), float(v[2])))
                if rows:
                    self.save_nli_batch(rows)
                    migrated += len(rows)
                nli_file.rename(nli_file.with_suffix(".json.migrated"))
            except Exception as e:
                print(f"[CacheDB] NLI migration: {e}")

        # --- Entity JSON ---
        ent_file = Path("cache/entity_sentences.json")
        if ent_file.exists():
            try:
                raw = json.loads(ent_file.read_text(encoding="utf-8"))
                sents, counts_l, strings_l = [], [], []
                for sent, payload in raw.items():
                    sents.append(sent)
                    counts_l.append(payload["counts"])
                    strings_l.append(payload["strings"])
                if sents:
                    self.save_entities_batch(sents, counts_l, strings_l)
                    migrated += len(sents)
                ent_file.rename(ent_file.with_suffix(".json.migrated"))
            except Exception as e:
                print(f"[CacheDB] Entity migration: {e}")

        # --- Relex triplets JSON ---
        rel_file = Path("cache/relex_triplets.json")
        if rel_file.exists():
            try:
                raw = json.loads(rel_file.read_text(encoding="utf-8"))
                sents = list(raw.keys())
                trips = [raw[s] for s in sents]
                if sents:
                    self.save_triplets_batch(sents, trips)
                    migrated += len(sents)
                rel_file.rename(rel_file.with_suffix(".json.migrated"))
            except Exception as e:
                print(f"[CacheDB] Relex migration: {e}")

        # --- Embedding .npz files ---
        embed_dir = Path("cache/embeddings")
        if embed_dir.exists():
            for npz_file in sorted(embed_dir.glob("*.npz")):
                try:
                    model_name = npz_file.stem.replace("__", "/")
                    data = np.load(npz_file, allow_pickle=True)
                    sents = data["sentences"].tolist()
                    embs = data["embeddings"]
                    self.save_embeddings_batch(
                        model_name, sents, [embs[i] for i in range(len(sents))]
                    )
                    migrated += len(sents)
                    npz_file.rename(npz_file.with_suffix(".npz.migrated"))
                except Exception as e:
                    print(f"[CacheDB] Embedding migration ({npz_file.name}): {e}")

        # --- Feature map JSON files (cache/{32-hex-char}.json) ---
        cache_dir = Path("cache")
        json_files = [
            f for f in cache_dir.glob("*.json")
            if len(f.stem) == 32 and all(c in "0123456789abcdef" for c in f.stem)
        ]
        if json_files:
            conn = self._conn()
            batch: list[tuple] = []
            to_rename: list[Path] = []
            for jf in json_files:
                try:
                    payload = json.loads(jf.read_text(encoding="utf-8"))
                    batch.append((jf.stem, "", "", json.dumps(payload, ensure_ascii=False)))
                    to_rename.append(jf)
                except Exception:
                    pass
            if batch:
                conn.executemany(
                    "INSERT OR IGNORE INTO features (md5, text1, text2, data) VALUES (?,?,?,?)",
                    batch,
                )
                conn.commit()
                migrated += len(batch)
                for jf in to_rename:
                    try:
                        jf.rename(jf.with_suffix(".json.migrated"))
                    except Exception:
                        pass

        if migrated:
            print(f"[CacheDB] migrated {migrated} entries from old file caches -> SQLite")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return row counts per table and database file size."""
        conn = self._conn()
        tables = ["embeddings", "nli_pairs", "efg_pairs", "entities", "triplets", "features"]
        counts = {}
        for t in tables:
            try:
                row = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
                counts[t] = row[0] if row else 0
            except Exception:
                counts[t] = 0
        size_mb = 0.0
        try:
            size_mb = round(self.db_path.stat().st_size / 1_048_576, 2)
        except Exception:
            pass
        return {"table_counts": counts, "db_size_mb": size_mb}
