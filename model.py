from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
from sklearn.base import BaseEstimator


MODEL_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "model.joblib")


@dataclass
class PredictionResult:
    label: str  
    probability: float  
    top_signals: List[Tuple[str, float]]  


class SpamDetector:
    def __init__(self, model_path: str = MODEL_PATH_DEFAULT):
        self.model_path = model_path
        self.pipeline: BaseEstimator | None = None
        self._feature_names: List[str] | None = None

        if os.path.exists(self.model_path):
            obj = joblib.load(self.model_path)
           
            if isinstance(obj, dict) and "pipeline" in obj:
                self.pipeline = obj["pipeline"]
                self._feature_names = obj.get("feature_names")
            else:
                self.pipeline = obj

    def ensure_loaded(self):
        if self.pipeline is None:
            raise RuntimeError(
                "Model not found. Run `python train.py` first to create model.joblib."
            )

    @staticmethod
    def extract_insight_signals(text: str) -> List[str]:
        """Heuristic signals that are useful for documentation/monitoring.

        These are not the only features used by the model; they are additional
        human-readable signals to make the output actionable.
        """
        t = text.lower()

        patterns = [
            ("urgent_terms", r"\b(urgent|asap|immediately|important|act now)\b"),
            ("free_terms", r"\b(free|guaranteed|no cost|prize|winner)\b"),
            ("financial_terms", r"\b(invest|investment|bank|wire|transfer|account|crypto|bitcoin|wallet)\b"),
            ("account_verification", r"\b(verify|verification|confirm|login|sign in|secure)\b"),
            ("threat_terms", r"\b(limited|suspended|blocked|frozen|overdue|penalty)\b"),
            ("malicious_link_indicators", r"(http[s]?://|www\.)"),
            ("attachments", r"\b(attachment|attached|invoice|document|pdf|docx)\b"),
            ("spoofing_phrases", r"\b(account team|customer support|security team|we noticed)\b"),
        ]

        found: List[str] = []
        for name, pat in patterns:
            if re.search(pat, t, flags=re.IGNORECASE):
                found.append(name)

        seen = set()
        out: List[str] = []
        for s in found:
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out

    def predict(self, email_text: str, *, top_k_signals: int = 8) -> PredictionResult:
        self.ensure_loaded()
        assert self.pipeline is not None

        proba = getattr(self.pipeline, "predict_proba")
        prob = float(proba([email_text])[0][1])  

        label = "spam" if prob >= 0.5 else "safe"

        top_signals: List[Tuple[str, float]] = []

        try:
            vectorizer = self.pipeline.named_steps["tfidf"]
            clf = self.pipeline.named_steps["clf"]
            if hasattr(vectorizer, "get_feature_names_out") and hasattr(clf, "coef_"):
                feature_names = self._feature_names or list(vectorizer.get_feature_names_out())
           
                coefs = clf.coef_
            
                coef_row = coefs[0] if coefs.shape[0] == 1 else coefs[1]

                x = self.pipeline.named_steps["tfidf"].transform([email_text])
         
                nz = x.nonzero()[1]
              
                scored: List[Tuple[str, float]] = []
                for idx in nz[:2000]:  
                    w = float(coef_row[idx])
                    if abs(w) < 0.0001:
                        continue
                    scored.append((feature_names[idx], w))

                scored.sort(key=lambda t: abs(t[1]), reverse=True)
                top_signals = scored[:top_k_signals]
        except Exception:
            top_signals = []

      
        heuristic = self.extract_insight_signals(email_text)
        if not top_signals:
      
            top_signals = [(h, (prob - 0.5) * 2.0) for h in heuristic[:top_k_signals]]
        else:
            for h in heuristic[: max(0, top_k_signals - len(top_signals))]:
                if all(sig != h for sig, _ in top_signals):
                    top_signals.append((h, (prob - 0.5) * 2.0))

        return PredictionResult(label=label, probability=prob, top_signals=top_signals)

