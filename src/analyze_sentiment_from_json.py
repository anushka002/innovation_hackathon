import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load transcript from JSON
def load_transcript(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['transcript']

# Analyze sentiment per segment
def analyze_sentiment(transcript_segments):
    analyzer = SentimentIntensityAnalyzer()
    analyzed = []

    for segment in transcript_segments:
        text = segment['text']
        sentiment = analyzer.polarity_scores(text)
        compound = sentiment['compound']

        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        analyzed.append({
            "start": segment['start'],
            "end": segment['end'],
            "text": text,
            "sentiment_score": round(compound * 100, 2),
            "sentiment_label": label
        })

    return analyzed

# Save result to JSON
def save_results(results, output_path="sentiment_results.json"):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved sentiment results to {output_path}")

# Run everything
if __name__ == "__main__":
    transcript = load_transcript("transcript_output.json")
    results = analyze_sentiment(transcript)
    save_results(results)
