from transformers import pipeline
import random

class NLPEngine:
    def __init__(self):
        self.sentiment_analyzer = None

    def _load_model(self):
        if self.sentiment_analyzer:
            return
        
        print("Loading NLP Model... (This may take a while)")
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english",
                top_k=None
            )
        except Exception as e:
            print(f"Warning: Transformers model failed to load. Using mock. {e}")
            self.sentiment_analyzer = "MOCK"

    def analyze_text_risk(self, texts):
        """
        Analyze a list of texts (commit messages, issues) and return a risk score.
        Negative sentiment -> Higher Risk
        """
        if not texts:
            return 0.0

        if not self.sentiment_analyzer:
            self._load_model()
            
        if self.sentiment_analyzer == "MOCK":
             return random.random() * 0.5


        try:
            # Analyze batch
            results = self.sentiment_analyzer(texts[:5]) # Limit to 5 for speed
            
            risk_score = 0.0
            count = 0
            for res in results:
                # res is a list of dicts [{'label': 'NEGATIVE', 'score': 0.9}, ...]
                # Find negative score
                neg_score = next((item['score'] for item in res if item['label'] == 'NEGATIVE'), 0.0)
                risk_score += neg_score
                count += 1
            
            return risk_score / count if count > 0 else 0.0
        except:
            return 0.5
