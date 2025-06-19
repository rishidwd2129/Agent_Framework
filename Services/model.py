# model.py
from transformers import pipeline

class SentimentAnalyzer:
    """
    A wrapper for a pre-trained sentiment analysis model from Hugging Face.
    This simulates a 'small custom Deep Learning model'.
    """
    def __init__(self):
        # Load a pre-trained sentiment analysis pipeline
        # 'distilbert-base-uncased-finetuned-sst-2-english' is a small, efficient model
        self.pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyzes the sentiment of the given text.
        Returns a dictionary with 'label' (POSITIVE/NEGATIVE) and 'score'.
        """
        if not text.strip():
            return {"label": "NEUTRAL", "score": 0.0}
        
        result = self.pipeline(text)[0]  # Added [0] to get the first result from the list
        return {"label": result['label'], "score": round(result['score'], 4)}

# Example usage (for testing this module independently)
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    print(analyzer.analyze_sentiment("I love this product! It's fantastic."))
    print(analyzer.analyze_sentiment("This is terrible, I hate it."))
    print(analyzer.analyze_sentiment("It's an okay day."))
    print(analyzer.analyze_sentiment(""))