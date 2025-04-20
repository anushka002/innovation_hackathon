import json
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def load_whisper_text(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return " ".join([chunk["text"] for chunk in data["transcription"]])

def get_word_contributions(sentence):
    words = sentence.split()
    word_scores = []

    for word in words:
        score = analyzer.polarity_scores(word)['compound']
        if score >= 0.1 or score <= -0.1:
            word_scores.append({
                "word": word,
                "sentiment_score": round(score, 2),
                "sentiment_label": "Positive" if score > 0 else "Negative"
            })

    return word_scores

def analyze_sentences_period_only(text):
    # Ultra-strict: only split on periods
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    results = []

    for i, sentence in enumerate(sentences):
        sentence += "."  # Add the period back
        sentiment = analyzer.polarity_scores(sentence)
        compound = sentiment["compound"]

        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        word_contributions = get_word_contributions(sentence)

        results.append({
            "sentence_number": i + 1,
            "text": sentence,
            "sentiment_score": round(compound * 100, 2),
            "sentiment_label": label,
            "contributing_words": word_contributions
        })

    return results

def save_results(results, output_path="sentiment_results.json"):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Saved detailed sentiment analysis to {output_path}")

def summarize_and_plot(results):
    sentence_nums = [r["sentence_number"] for r in results]
    scores = [r["sentiment_score"] for r in results]
    labels = [r["sentiment_label"] for r in results]

    avg_score = sum(scores) / len(scores)
    positive = labels.count("Positive")
    neutral = labels.count("Neutral")
    negative = labels.count("Negative")
    total = len(labels)

    pos_words = []
    neg_words = []

    for r in results:
        for word_info in r["contributing_words"]:
            if word_info["sentiment_label"] == "Positive":
                pos_words.append(word_info["word"])
            elif word_info["sentiment_label"] == "Negative":
                neg_words.append(word_info["word"])

    top_pos = Counter(pos_words).most_common(5)
    top_neg = Counter(neg_words).most_common(5)

    best_sent = max(results, key=lambda x: x["sentiment_score"])
    worst_sent = min(results, key=lambda x: x["sentiment_score"])

    summary = {
        "average_sentiment_score": round(avg_score, 2),
        "positive_sentences": positive,
        "neutral_sentences": neutral,
        "negative_sentences": negative,
        "total_sentences": total,
        "positivity_percentage": round((positive / total) * 100, 2),
        "top_positive_contributing_words": [{"word": w, "count": c} for w, c in top_pos],
        "top_negative_contributing_words": [{"word": w, "count": c} for w, c in top_neg],
        "best_sentence": {
            "sentence_number": best_sent["sentence_number"],
            "text": best_sent["text"],
            "score": best_sent["sentiment_score"]
        },
        "worst_sentence": {
            "sentence_number": worst_sent["sentence_number"],
            "text": worst_sent["text"],
            "score": worst_sent["sentiment_score"]
        },
        "insight": f"Sentence {worst_sent['sentence_number']} made the sentiment score lower, while Sentence {best_sent['sentence_number']} had the most positive tone overall."
    }

    with open("sentiment_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("✅ Summary saved to sentiment_summary.json")

    # Plot sentiment over time
    plt.figure(figsize=(10, 5))
    plt.plot(sentence_nums, scores, marker='o', linewidth=2)
    plt.title("Sentence-Level Sentiment Over Time (Strict Split)")
    plt.xlabel("Sentence #")
    plt.ylabel("Sentiment Score (Compound x 100)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sentiment_plot.png")
    plt.close()
    print("✅ Sentiment plot saved as sentiment_plot.png")

if __name__ == "__main__":
    text = load_whisper_text("whisper_transcription.json")  # Your input JSON
    results = analyze_sentences_period_only(text)
    save_results(results)
    summarize_and_plot(results)
