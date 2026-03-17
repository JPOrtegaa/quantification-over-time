from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
# Preprocess text (username and link placeholders)


import torch


class SentiAnalyzer:
    def __init__(self, model):

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

    def batch_analyze(self, texts, batch_size=32):
        all_scores = []
        all_labels = []

        for i in range(0, len(texts), batch_size):
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
