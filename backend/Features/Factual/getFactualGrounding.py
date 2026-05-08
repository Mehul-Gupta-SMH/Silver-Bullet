import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from backend.Postprocess.__addpad import resize_matrix
from backend.cache_db import CacheDB

_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


class EFGGrounding:
    """External Factual Grounding — DeBERTa-FEVER cross-sentence factual support scores.

    Produces three n×m feature maps:
      efg_supports      — P(s2_j is factually supported by s1_i), forward direction
      efg_refutes       — P(s2_j is factually refuted by s1_i), forward direction
      efg_factual_delta — efg_supports(s1→s2) minus efg_supports(s2→s1); positive
                          values indicate s1 is more factually authoritative than s2

    Distinct from NLIWeights (roberta-large-mnli): the FEVER fine-tuning makes
    this model sensitive to factual accuracy rather than pure semantic entailment.
    """

    _model_cache: dict = {}
    _pair_cache: dict = {}   # (sent1, sent2) → (supports, refutes, nei)
    _disk_cache_loaded: bool = False

    def __init__(self, model_name: str = _MODEL_NAME):
        self.model_name = model_name
        self.__batch_size__ = 32
        self.__max_len__ = 128
        self._reset_state()

    def _reset_state(self):
        self.phrase_list1 = None
        self.phrase_list2 = None
        self.comparison_weights = {}

    # ------------------------------------------------------------------
    # Persistent pair cache
    # ------------------------------------------------------------------

    @classmethod
    def load_pair_cache(cls) -> None:
        if cls._disk_cache_loaded:
            return
        cls._disk_cache_loaded = True
        try:
            rows = CacheDB.get().load_all_efg()
            cls._pair_cache.update(rows)
            print(f"[EFGCache] loaded {len(rows)} persisted sentence-pair scores from SQLite")
        except Exception as e:
            print(f"[EFGCache] warning: could not load persistent cache ({e}) — starting fresh")

    @classmethod
    def save_pair_cache(cls, new_rows: list) -> None:
        if not new_rows:
            return
        try:
            CacheDB.get().save_efg_batch(new_rows)
        except Exception as e:
            print(f"[EFGCache] warning: could not save to SQLite ({e})")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def __load_model__(self):
        tok = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=f"Features/Factual/model/{self.model_name}/",
            model_max_length=self.__max_len__,
        )
        mdl = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            cache_dir=f"Features/Factual/model/{self.model_name}/",
            low_cpu_mem_usage=False,
        )
        mdl.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(device)
        EFGGrounding._model_cache[self.model_name] = {
            "tokenizer": tok,
            "model": mdl,
            "device": device,
        }

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def __calc_weights__(self):
        EFGGrounding.load_pair_cache()

        # Collect all uncached directional pairs (both forward and backward needed
        # for the delta map).
        all_pairs: set[tuple[str, str]] = set()
        for s1 in self.phrase_list1:
            for s2 in self.phrase_list2:
                if (s1, s2) not in EFGGrounding._pair_cache:
                    all_pairs.add((s1, s2))
                if (s2, s1) not in EFGGrounding._pair_cache:
                    all_pairs.add((s2, s1))

        if all_pairs:
            if self.model_name not in EFGGrounding._model_cache:
                self.__load_model__()

            mc = EFGGrounding._model_cache[self.model_name]
            tok, mdl, device = mc["tokenizer"], mc["model"], mc["device"]

            id2label = mdl.config.id2label
            label_to_idx = {v.lower(): k for k, v in id2label.items()}
            ent_idx = label_to_idx.get("entailment", 0)
            ref_idx = label_to_idx.get("contradiction", 2)
            nei_idx = label_to_idx.get("neutral", 1)

            uncached = list(all_pairs)
            new_rows: list[tuple] = []
            with torch.no_grad():
                for start in range(0, len(uncached), self.__batch_size__):
                    chunk = uncached[start : start + self.__batch_size__]
                    enc = tok(
                        [p for p, _ in chunk],
                        [h for _, h in chunk],
                        padding=True,
                        truncation=True,
                        max_length=self.__max_len__,
                        return_tensors="pt",
                    ).to(device)
                    logits = mdl(**enc).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    for (s1, s2), row in zip(chunk, probs):
                        scores = (float(row[ent_idx]), float(row[ref_idx]), float(row[nei_idx]))
                        EFGGrounding._pair_cache[(s1, s2)] = scores
                        new_rows.append((s1, s2, scores[0], scores[1], scores[2]))

            EFGGrounding.save_pair_cache(new_rows)

        # Reconstruct n×m matrices from cache.
        fwd_sup, fwd_ref, delta = [], [], []
        _default = (1 / 3, 1 / 3, 1 / 3)
        for s1 in self.phrase_list1:
            row_sup, row_ref, row_delta = [], [], []
            for s2 in self.phrase_list2:
                sup_fwd, ref_fwd, _ = EFGGrounding._pair_cache.get((s1, s2), _default)
                sup_bwd, _, _ = EFGGrounding._pair_cache.get((s2, s1), _default)
                row_sup.append(sup_fwd)
                row_ref.append(ref_fwd)
                row_delta.append(sup_fwd - sup_bwd)
            fwd_sup.append(row_sup)
            fwd_ref.append(row_ref)
            delta.append(row_delta)

        self.comparison_weights = {
            "efg_supports": fwd_sup,
            "efg_refutes": fwd_ref,
            "efg_factual_delta": delta,
        }

    def __post_process_weights__(self):
        for key in self.comparison_weights:
            self.comparison_weights[key] = resize_matrix(self.comparison_weights[key])

    def getFeatureMap(self, phrase_list1: list, phrase_list2: list) -> dict:
        self._reset_state()
        self.phrase_list1, self.phrase_list2 = phrase_list1, phrase_list2
        self.__calc_weights__()
        self.__post_process_weights__()
        return self.comparison_weights
