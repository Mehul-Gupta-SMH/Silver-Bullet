import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from backend.Postprocess.__addpad import resize_matrix

_NLI_CACHE_FILE = Path("cache/nli_pairs.json")


class NLIWeights:
    # Shared across all instances — model loaded once per process
    _model_cache: dict = {}
    # Sentence-pair NLI cache: (sent1, sent2) → (entailment, neutral, contradiction)
    # Persisted to cache/nli_pairs.json between runs so RoBERTa is never re-run
    # for a sentence pair that has already been scored.
    _pair_cache: dict = {}
    _disk_cache_loaded: bool = False

    def __init__(self, MODEL_NAME: str = "FacebookAI/roberta-large-mnli"):
        self.ModelName = MODEL_NAME
        self.__batch_size__ = 64
        self.__max_len__ = 256
        self._reset_state()

    def _reset_state(self):
        """Reset per-call accumulation buffers. Preserves the loaded model cache."""
        self.phrase_list1 = None
        self.phrase_list2 = None
        self.comparison_weights = {}

    # ------------------------------------------------------------------
    # Persistent pair cache — load / save
    # ------------------------------------------------------------------

    @classmethod
    def load_pair_cache(cls) -> None:
        """Load persisted NLI pair scores from disk into _pair_cache.

        Called once per process before the first scoring pass. Subsequent calls
        are no-ops (guarded by _disk_cache_loaded). This means once a sentence
        pair is scored it is never re-run, even across separate training runs.
        """
        if cls._disk_cache_loaded:
            return
        cls._disk_cache_loaded = True
        if not _NLI_CACHE_FILE.exists():
            return
        try:
            raw = json.loads(_NLI_CACHE_FILE.read_text(encoding="utf-8"))
            for k, v in raw.items():
                s1, s2 = json.loads(k)
                cls._pair_cache[(s1, s2)] = tuple(v)
            print(f"[NLICache] loaded {len(cls._pair_cache)} persisted sentence-pair scores from disk")
        except Exception as e:
            print(f"[NLICache] warning: could not load persistent cache ({e}) — starting fresh")

    @classmethod
    def save_pair_cache(cls) -> None:
        """Persist current _pair_cache to disk.

        Uses json.dumps([s1, s2]) as the key so there is no ambiguity from
        separator characters inside sentences.
        """
        _NLI_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {
            json.dumps([s1, s2]): list(v)
            for (s1, s2), v in cls._pair_cache.items()
        }
        _NLI_CACHE_FILE.write_text(
            json.dumps(serialisable, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------

    def __load_model__(self):
        tok = AutoTokenizer.from_pretrained(
            self.ModelName,
            cache_dir=f"Features/NLI/model/{self.ModelName}/",
            model_max_length=self.__max_len__,  # silences truncation warning
        )
        # low_cpu_mem_usage=False: prevents meta-tensor initialisation that
        # makes .to(device) raise "Cannot copy out of meta tensor".
        mdl = AutoModelForSequenceClassification.from_pretrained(
            self.ModelName,
            cache_dir=f"Features/NLI/model/{self.ModelName}/",
            low_cpu_mem_usage=False,
        )
        mdl.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(device)
        NLIWeights._model_cache[self.ModelName] = {
            "tokenizer": tok,
            "model": mdl,
            "device": device,
        }

    def __calc_weights__(self):
        # Ensure the persistent cache is loaded (no-op after first call).
        NLIWeights.load_pair_cache()

        # Collect all (p1, p2) pairs that are not yet in the pair cache.
        uncached = [
            (p1, p2)
            for p1 in self.phrase_list1
            for p2 in self.phrase_list2
            if (p1, p2) not in NLIWeights._pair_cache
        ]

        if uncached:
            if self.ModelName not in NLIWeights._model_cache:
                self.__load_model__()

            mc = NLIWeights._model_cache[self.ModelName]
            tok, mdl, device = mc["tokenizer"], mc["model"], mc["device"]
            id2label = mdl.config.id2label
            label_to_idx = {v.lower(): k for k, v in id2label.items()}

            with torch.no_grad():
                for start in range(0, len(uncached), self.__batch_size__):
                    chunk = uncached[start:start + self.__batch_size__]
                    enc = tok(
                        [p for p, _ in chunk],
                        [h for _, h in chunk],
                        padding=True, truncation=True,
                        max_length=self.__max_len__,
                        return_tensors="pt",
                    ).to(device)
                    logits = mdl(**enc).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    for (p1, p2), row in zip(chunk, probs):
                        NLIWeights._pair_cache[(p1, p2)] = (
                            float(row[label_to_idx["entailment"]]),
                            float(row[label_to_idx["neutral"]]),
                            float(row[label_to_idx["contradiction"]]),
                        )

            # Persist new scores immediately so they survive even if training
            # is interrupted before completing.
            NLIWeights.save_pair_cache()

        # Reconstruct the n×m matrices from the cache (no model call needed).
        row_e_all, row_n_all, row_c_all = [], [], []
        for p1 in self.phrase_list1:
            row_e, row_n, row_c = [], [], []
            for p2 in self.phrase_list2:
                e, n, c = NLIWeights._pair_cache[(p1, p2)]
                row_e.append(e)
                row_n.append(n)
                row_c.append(c)
            row_e_all.append(row_e)
            row_n_all.append(row_n)
            row_c_all.append(row_c)

        self.comparison_weights = {
            "entailment":     row_e_all,
            "neutral":        row_n_all,
            "contradiction":  row_c_all,
        }

    def __post_process_weights__(self):
        for key in self.comparison_weights:
            self.comparison_weights[key] = resize_matrix(self.comparison_weights[key])

    def getFeatureMap(self, phrase_list1: list, phrase_list2: list):
        self._reset_state()
        self.phrase_list1, self.phrase_list2 = phrase_list1, phrase_list2
        self.__calc_weights__()
        self.__post_process_weights__()
        return self.comparison_weights


if __name__ == "__main__":
    sample_text = [
        'My Name is Mehul',
        'Mehul is a good person',
        'Mehul is a bad person'
    ]
    nli = NLIWeights()
    print(nli.getFeatureMap(sample_text, sample_text))
