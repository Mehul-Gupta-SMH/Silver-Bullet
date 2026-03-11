import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from Postprocess.__addpad import pad_matrix


class NLIWeights:
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

    def __load_model__(self):
        tok = AutoTokenizer.from_pretrained(
            self.ModelName,
            cache_dir=f"Features/NLI/model/{self.ModelName}/"
        )
        mdl = AutoModelForSequenceClassification.from_pretrained(
            self.ModelName,
            cache_dir=f"Features/NLI/model/{self.ModelName}/"
        )
        mdl.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(device)
        self.__model_cache__ = {
            "tokenizer": tok,
            "model": mdl,
            "device": device
        }

    def __calc_weights__(self):
        if not hasattr(self, "__model_cache__"):
            self.__load_model__()

        tok = self.__model_cache__["tokenizer"]
        mdl = self.__model_cache__["model"]
        device = self.__model_cache__["device"]

        id2label = mdl.config.id2label
        label_to_idx = {v.lower(): k for k, v in id2label.items()}

        row_e_all, row_n_all, row_c_all = [], [], []

        with torch.no_grad():
            for p1 in tqdm(self.phrase_list1, desc="NLI Weights - Phrase 1"):
                row_e, row_n, row_c = [], [], []
                pairs = [(p1, p2) for p2 in self.phrase_list2]

                for start in range(0, len(pairs), self.__batch_size__):
                    chunk = pairs[start:start + self.__batch_size__]
                    enc = tok(
                        [p for p, _ in chunk],
                        [h for _, h in chunk],
                        padding=True, truncation=True,
                        max_length=self.__max_len__,
                        return_tensors="pt"
                    ).to(device)

                    logits = mdl(**enc).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()

                    row_e.extend(probs[:, label_to_idx["entailment"]].tolist())
                    row_n.extend(probs[:, label_to_idx["neutral"]].tolist())
                    row_c.extend(probs[:, label_to_idx["contradiction"]].tolist())

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
            self.comparison_weights[key] = pad_matrix(self.comparison_weights[key])

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
