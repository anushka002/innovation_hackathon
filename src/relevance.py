from ollama import chat, ChatResponse
import re
import json

# --- Ollama Evaluation Function ---
def evaluate_answer(question, answer, model="llama3.2"):
    prompt = f"""You are an interview evaluator.

Question: {question}

Answer: {answer}

Evaluate how well the answer responds to the question.

Respond in the following format:
Score: <numeric score between 0 and 100>

Explanation: <brief justification>"""

    try:
        response: ChatResponse = chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        return response.message.content
    except Exception as e:
        return f"Error: {e}"

# --- Score & Observation Extraction ---
def extract_score_and_observation(text):
    # Match formats like "Score: 85", "score is 90", "score of 75"
    score_match = re.search(r"score(?:[:\s]*| is | of )(\d{1,3})", text, re.IGNORECASE)
    score = int(score_match.group(1)) if score_match else None

    # Remove the score line for cleaner explanation
    observation = re.sub(r"score(?:[:\s]*| is | of )\d{1,3}", "", text, flags=re.IGNORECASE).strip()
    return {
        "score": score,
        "observation": observation
    }

# --- Main ---
if __name__ == "__main__":
    question = "Tell me about a time you disagreed with your boss / Supervisor."

    # Read answer from file
    try:
        with open("whisper_transcription.txt", "r", encoding="utf-8") as f:
            answer = f.read().strip()
    except FileNotFoundError:
        print("❌ Error: 'whisper_transcription.txt' not found.")
        exit(1)

    # Call Ollama
    result = evaluate_answer(question, answer)
    print("\nRaw Output:\n", result)

    # Extract score + summary
    output = extract_score_and_observation(result)
    print("\nJSON Output:\n", json.dumps(output, indent=2))

    # Save to file
    with open("evaluation_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n✅ Result saved to 'evaluation_output.json'")
