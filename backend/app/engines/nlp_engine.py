import os


class NLPEngine:
    def __init__(self):
        self.sentiment_analyzer = None
        self.model_type = None
        self.model_path = os.path.join(os.path.dirname(__file__), "nlp_model.joblib")

    def _load_model(self):
        # MEMORY OPTIMIZATION:
        # Default to 'mock' or 'sklearn' (Lite Mode) to save RAM on Render (512MB limit).
        # Only load Transformers if explicitly requested via env var or if resources allow.
        
        enable_transformers = os.getenv("ENABLE_TRANSFORMERS", "false").lower() == "true"
        
        if enable_transformers:
            try:
                from transformers import pipeline
                self.sentiment_analyzer = pipeline(
                    "text-classification",
                    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                    top_k=None,
                )
                self.model_type = 'transformers'
                print("NLPEngine: loaded transformers pipeline.")
                return
            except Exception as e:
                print(f"NLPEngine: Transformers load failed: {e}")

        # Try sklearn fallback (load or train a tiny model)
        try:
            from joblib import load
            if os.path.exists(self.model_path):
                self.sentiment_analyzer = load(self.model_path)
                self.model_type = 'sklearn'
                print(f"NLPEngine: loaded sklearn model from {self.model_path}")
                return
        except Exception:
            pass

        # If sklearn model not found, we'll train one on demand when needed.
        self.model_type = None

    def _train_default_sklearn(self):
        try:
            from sklearn.pipeline import make_pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from joblib import dump

            risky = [
                "exploit", "vulnerability", "CVE", "buffer overflow", "privilege escalation",
                "unauthenticated access", "bypass authentication", "race condition", "memory leak"
            ]
            safe = [
                "update docs", "refactor code", "add tests", "improve performance", "minor fix",
                "chore", "cleanup", "enhancement", "typo fix"
            ]

            X = risky + safe
            y = [1] * len(risky) + [0] * len(safe)

            pipeline = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), max_features=2000),
                LogisticRegression(max_iter=1000)
            )
            pipeline.fit(X, y)

            dump(pipeline, self.model_path)
            self.sentiment_analyzer = pipeline
            self.model_type = 'sklearn'
            print(f"NLPEngine: trained and saved sklearn model to {self.model_path}")
            return True
        except Exception as e:
            print(f"NLPEngine: sklearn training failed: {e}")
            return False

    def _ensure_sklearn_loaded(self):
        # Attempt to load existing model, else train a minimal one
        try:
            from joblib import load
            if os.path.exists(self.model_path):
                self.sentiment_analyzer = load(self.model_path)
                self.model_type = 'sklearn'
                return True
        except Exception:
            pass
        return self._train_default_sklearn()

    def analyze_text_risk(self, texts):
        """Analyze texts and return a risk score in [0,1]."""
        if not texts:
            return 0.0

        if isinstance(texts, str):
            texts = [texts]

        # Ensure a model is loaded
        if not self.model_type:
            # Try to load transformers/sklearn or train sklearn
            self._load_model()
            if not self.model_type:
                # try ensure sklearn (train) if transformers missing
                if not self._ensure_sklearn_loaded():
                    self.model_type = 'mock'

        if self.model_type == 'transformers' and self.sentiment_analyzer:
            try:
                results = self.sentiment_analyzer(texts[:5])
                scores = []
                for r in results:
                    if isinstance(r, list):
                        neg = next((c['score'] for c in r if c.get('label', '').upper().startswith('NEG')), 0.0)
                        scores.append(neg)
                    else:
                        label = r.get('label', '').upper()
                        score = r.get('score', 0.0)
                        scores.append(score if label.startswith('NEG') else (1 - score))
                return float(sum(scores) / max(1, len(scores)))
            except Exception as e:
                print(f"NLPEngine: transformers analysis failed: {e}")

        if self.model_type == 'sklearn' and self.sentiment_analyzer:
            try:
                probs = self.sentiment_analyzer.predict_proba(texts[:32])
                risky_probs = [p[1] for p in probs]
                return float(sum(risky_probs) / max(1, len(risky_probs)))
            except Exception as e:
                print(f"NLPEngine: sklearn analysis failed: {e}")

        # Final fallback
        return 0.25
