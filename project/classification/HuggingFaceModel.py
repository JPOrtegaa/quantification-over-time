from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from tqdm import tqdm

import config
# Preprocess text (username and link placeholders)


import torch


class SentiAnalyzer:
    def __init__(self, model):
        self._model_id = model
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.config = AutoConfig.from_pretrained(model)

        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    """
    here we can see if different preprocessing 
    """

    def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            new_text.append(t)

        return " ".join(new_text)

    def analyze(self, text):
        text = SentiAnalyzer.preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**encoded_input)
        scores = output[0][0].detach().cpu().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        l = self.config.id2label[ranking[0]]

        return scores, l

    def batch_analyze(
        self,
        texts,
        batch_size=None,
        show_progress=True,
        hf_context=None,
        announce=True,
    ):
        if batch_size is None:
            batch_size = getattr(config, "HF_INFERENCE_BATCH_SIZE", 8)
        all_scores = []
        all_labels = []

        n = len(texts)
        num_batches = (n + batch_size - 1) // batch_size if n else 0
        ctx = (hf_context or "HF sentiment forward").strip().replace("\n", " ")
        if len(ctx) > 72:
            ctx = ctx[:69] + "…"
        if announce and n > 0:
            print(
                f"[HF] {ctx} | texts={n} batches={num_batches}",
                flush=True,
            )
        batch_indices = range(0, n, batch_size)
        if show_progress and num_batches > 0:
            batch_indices = tqdm(
                batch_indices,
                total=num_batches,
                desc="HF",
                unit="batch",
                leave=False,
            )

        for i in batch_indices:
            batch_texts = texts[i : i + batch_size]
            batch_texts = [SentiAnalyzer.preprocess(str(x)) for x in batch_texts]
            encoded_input = self.tokenizer(
                batch_texts, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                output = self.model(**encoded_input)
            batch_scores = output[0].detach().cpu().numpy()
            batch_scores = softmax(batch_scores, axis=1)

            for scores in batch_scores:
                ranking = np.argsort(scores)
                ranking = ranking[::-1]
                label = self.config.id2label[ranking[0]]
                all_scores.append(scores)
                all_labels.append(label)

        return np.array(all_scores), all_labels
