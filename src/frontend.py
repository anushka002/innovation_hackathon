import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import random
import shutil
import pandas as pd
import json
import os
import io
import base64
import time

# Function to load JSON files
def load_json_file(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found. Please ensure it exists.")
        with open(file_path, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding {file_path}: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading {file_path}: {str(e)}")

# Construct absolute paths to JSON files
script_dir = os.path.dirname(os.path.abspath(__file__))
sentiment_file = os.path.join(script_dir, "sentiment.json")
relevance_file = os.path.join(script_dir, "relevance.json")

# Load sentiment and relevance data
try:
    sentiment_data = load_json_file(sentiment_file)
    relevance_data = load_json_file(relevance_file)
except Exception as e:
    print(f"Error: {str(e)}")
    exit(1)

# === Practice Questions ===
questions_practice_mode = [
    "Tell me about yourself.",
    "Why do you want this role?",
    "Describe a time you handled a challenge at work.",
    "What are your greatest strengths?",
    "What is your biggest weakness?",
    "Where do you see yourself in 5 years?",
    "Why should we hire you?",
    "How do you handle pressure and deadlines?",
    "Give an example of teamwork in your past roles.",
    "What motivates you to perform well?"
]

# === Interview Questions ===
interview_questions_list = [
    "What excites you about working at TechNova Inc?",
    "How would you handle a situation where a project deadline is at risk?",
    "Describe a technical decision you made and what trade-offs were involved.",
    "How do you stay updated with current trends in our industry?",
    "Walk me through how you would approach leading a new team."
]

def get_random_question():
    return random.choice(questions_practice_mode)

def dummy_submit(video, question):
    # Extract data from JSON
    sentiment = round(sentiment_data.get("average_sentiment_score", 0) / 10)  # Scale to 0-10
    relevance = round(relevance_data.get("score", 0) / 10)  # Scale to 0-10
    final_score = round((sentiment + relevance) / 2)
    score = f"{final_score}/10"
    rank = "Top 5% applicant" if final_score >= 8 else "Needs improvement"

    final_feedback = (
        "Excellent performance. You're well-prepared and articulate." if final_score >= 9 else
        "Strong answers with minor areas to polish. Keep practicing." if final_score >= 7 else
        "Fair attempt, but consider refining your answers and delivery." if final_score >= 5 else
        "Your responses need improvement in clarity and relevance."
    )

    sentiment_feedback = {
        10: "You conveyed your message with confidence and warmth. Great job!",
        9: "Confident and friendly tone throughout your answer.",
        8: "Clear and warm delivery. Slightly more variation could help.",
        7: "Good tone but felt a bit flat at times.",
        6: "Delivery was calm, but lacked expressiveness.",
        5: "Seemed slightly nervous. Try to relax and smile naturally.",
        4: "Tone felt robotic or disconnected. Bring more energy.",
        3: "Low confidence detected in your voice.",
        2: "You sounded very unsure. Practice speaking clearly.",
        1: "Nervous and unclear. Mock interviews will help a lot."
    }.get(sentiment, "Sentiment feedback unavailable.")

    relevance_feedback = {
        10: "Your answer directly addressed the question with full clarity.",
        9: "Highly relevant with detailed explanation.",
        8: "Good response — mostly on track.",
        7: "Relevant, but had minor digressions.",
        6: "Wavered slightly off-topic at moments.",
        5: "General response. Lacked clear link to the question.",
        4: "Somewhat vague and indirect.",
        3: "Did not clearly address the core of the question.",
        2: "Mostly off-topic — revise your understanding of the question.",
        1: "Unrelated answer — make sure to listen/read carefully."
    }.get(relevance, "Relevance feedback unavailable.")

    # Create pie chart visualization with black background and white text
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), facecolor='black')
    fig.patch.set_facecolor('black')

    # Sentiment pie chart
    ax1.pie([sentiment, 10 - sentiment], labels=[f"{sentiment}/10", "Missed"],
            colors=['#f497a9', '#f2f2f2'], startangle=90, autopct='%1.0f%%',
            shadow=True, explode=[0.05, 0], textprops={'fontsize': 9, 'color': 'white'})
    ax1.set_title("Sentiment Score", fontsize=12, color='white')
    ax1.set_facecolor('black')

    # Relevance pie chart
    ax2.pie([relevance, 10 - relevance], labels=[f"{relevance}/10", "Missed"],
            colors=['#72d6c9', '#eeeeee'], startangle=90, autopct='%1.0f%%',
            shadow=True, explode=[0.05, 0], textprops={'fontsize': 9, 'color': 'white'})
    ax2.set_title("Relevance Score", fontsize=12, color='white')
    ax2.set_facecolor('black')

    plt.suptitle("Score Breakdown", fontsize=13, color='white')
    plt.tight_layout()

    # Create score table
    score_table_data = [
        {"Metric": "Relevance Score", "Value": f"{relevance}/10", "Feedback": relevance_feedback},
        {"Metric": "Sentiment Score", "Value": f"{sentiment}/10", "Feedback": sentiment_feedback}
    ]
    score_table_df = pd.DataFrame(score_table_data)
    score_table_html = score_table_df.to_html(index=False, classes="score-table", justify="left")

    # Create insights text
    insights = (
        f"<b>Insights:</b><br>"
        f"<b>Relevance Observation:</b> {relevance_data.get('observation', 'No observation available.')}<br>"
        f"<b>Sentiment Insight:</b> {sentiment_data.get('insight', 'No insight available.')}"
    )

    # Create word table without row coloring
    pos_df = pd.DataFrame(sentiment_data.get("top_positive_contributing_words", []))
    neg_df = pd.DataFrame(sentiment_data.get("top_negative_contributing_words", []))
    pos_df["Type"] = "Positive"
    neg_df["Type"] = "Negative"
    combined_df = pd.concat([pos_df, neg_df], ignore_index=True)[["Type", "word", "count"]]
    word_table_html = combined_df.to_html(index=False, classes="word-table", justify="center")

    # Create word cloud with separate colors for positive (greenish) and negative (reddish) words
    pos_words = {item["word"]: item["count"] for item in sentiment_data.get("top_positive_contributing_words", [])}
    neg_words = {item["word"]: item["count"] for item in sentiment_data.get("top_negative_contributing_words", [])}
    wordcloud_html = ""
    if pos_words or neg_words:
        # Create a figure with increased size (1.5x: 400x200 -> 600x300)
        plt.figure(figsize=(6, 3), facecolor='black')
        
        # Generate positive word cloud (greenish shades)
        if pos_words:
            wordcloud_pos = WordCloud(
                width=600, height=300,
                background_color=None,  # Transparent background for overlay
                mode="RGBA",
                min_font_size=8, max_font_size=40,
                colormap='Greens'  # Greenish shades
            ).generate_from_frequencies(pos_words)
            plt.imshow(wordcloud_pos, interpolation='bilinear')
        
        # Generate negative word cloud (reddish shades) and overlay
        if neg_words:
            wordcloud_neg = WordCloud(
                width=600, height=300,
                background_color=None,  # Transparent background for overlay
                mode="RGBA",
                min_font_size=8, max_font_size=40,
                colormap='Reds'  # Reddish shades
            ).generate_from_frequencies(neg_words)
            plt.imshow(wordcloud_neg, interpolation='bilinear')
        
        plt.axis('off')
        plt.tight_layout()
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        wordcloud_html = f'<img src="data:image/png;base64,{img_base64}" style="max-width: 600px; height: auto;">'
    else:
        wordcloud_html = '<p>No words available for word cloud.</p>'

    return (
        score, rank, fig,
        score_table_html, insights, word_table_html, wordcloud_html
    )

def save_video(video_path, question):
    if video_path:
        shutil.copy(video_path, "input.mp4")
    return dummy_submit(video_path, question)

def run_interview(index, time_left, timer_active, skip=False):
    # Handle skipping to the next question
    if skip:
        time_left = 90  # Reset timer
        timer_active = True  # Keep timer active for the next question
        index += 1

    # Check if interview is complete
    if index >= len(interview_questions_list):
        return (
            "<div class='question-box'>✅ Interview complete. Thank you!</div>",
            index,
            "",
            gr.update(visible=False),
            0,
            False,
            False  # Deactivate timer
        )

    # Start or continue timer
    if not timer_active:
        timer_active = True
        time_left = 90

    # Update question
    q = f"<div class='question-box'><strong>Q{index+1}:</strong> {interview_questions_list[index]}</div>"

    # Update timer
    if time_left > 0:
        mins, secs = divmod(time_left, 60)
        timer_text = f"Time Left: {mins}:{secs:02d}"
        time_left -= 1
    else:
        # Timer finished, move to next question
        timer_active = True  # Keep timer active for the next question
        index += 1
        time_left = 90
        if index < len(interview_questions_list):
            q = f"<div class='question-box'><strong>Q{index+1}:</strong> {interview_questions_list[index]}</div>"
            mins, secs = divmod(time_left, 60)
            timer_text = f"Time Left: {mins}:{secs:02d}"
        else:
            return (
                "<div class='question-box'>✅ Interview complete. Thank you!</div>",
                index,
                "",
                gr.update(visible=False),
                0,
                False,
                False  # Deactivate timer
            )

    return (
        q,
        index,
        timer_text,
        gr.update(visible=True),
        time_left,
        timer_active,
        True  # Keep timer active
    )

def start_interview(index, time_left, timer_active):
    # Reset states when starting the interview
    return (
        "<div class='question-box'><strong>Q1:</strong> {}</div>".format(interview_questions_list[0]),
        0,
        "Time Left: 1:30",
        gr.update(visible=True),
        90,
        True,
        True  # Activate timer
    )

def reset_ui():
    return get_random_question(), None, "", "", "", "", ""

# === UI Starts Here ===
with gr.Blocks(css="""
  .question-box {
    background-color: #2e2e2e;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #555;
    margin-bottom: 10px;
  }
  .green-button {
    background-color: #4CAF50 !important;
    color: white !important;
  }
  .red-button {
    background-color: #f44336 !important;
    color: white !important;
  }
  .blue-button {
    background-color: #1976D2 !important;
    color: white !important;
  }
  .video-practice video {
    max-width: 600px !important;
    height: auto !important;
    border-radius: 6px;
    border: 1px solid #666;
  }
  .video-interview video {
    max-width: 600px !important;
    height: auto !important;
    border-radius: 6px;
    border: 1px solid #666;
  }
  .score-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
  }
  .score-table th, .score-table td {
    border: 1px solid #ddd;
    padding: 10px;
    text-align: left;
  }
  .score-table th {
    background-color: #e0e0e0;
    color: #333;
    font-weight: bold;
  }
  .word-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
    background-color: #000;
    color: #fff;
  }
  .word-table th, .word-table td {
    border: 1px solid #555;
    padding: 8px;
    text-align: center;
    color: #fff !important;
  }
  .word-table th {
    background-color: #222;
    color: #fff;
    font-weight: bold;
  }
  .score-box {
    font-size: 32px !important;
    font-weight: bold;
    color: #333;
    margin-top: 10px;
  }
""") as demo:

    question_state = gr.State(get_random_question())
    interview_index = gr.State(0)
    time_left = gr.State(90)
    timer_active = gr.State(False)
    timer_state = gr.State(False)

    with gr.Tabs():
        with gr.Tab("Practice Mode"):
            gr.Markdown("## Practice Mode – Interview Prep")
            gr.Markdown("### Prepare with Purpose: Practice, Record, Improve")

            with gr.Row():
                with gr.Column(scale=1):
                    question_display = gr.Markdown(
                        value=f"<div class='question-box'><strong>Your Question:</strong><br>{get_random_question()}</div>"
                    )
                    video_input = gr.Video(elem_classes="video-practice")

                    with gr.Row(equal_height=True):
                        refresh_btn = gr.Button("🔄 New Question")
                        record_btn = gr.Button("▶ Record")
                        submit_btn = gr.Button("Submit", elem_classes="green-button")
                        reset_btn = gr.Button("Reset", elem_classes="red-button")

                    gr.Markdown("### Final Evaluation")
                    score_box = gr.Textbox(label="Overall Score", interactive=False, elem_classes="score-box")
                    rank_box = gr.Textbox(label="Candidate Rank", interactive=False)
                    final_feedback_box = gr.Markdown("")

                with gr.Column(scale=1):
                    gr.Markdown("### Visual Breakdown")
                    report_plot = gr.Plot()
                    gr.Markdown("#### Score Details")
                    score_table = gr.HTML()
                    gr.Markdown("#### Insights")
                    insights = gr.HTML()
                    gr.Markdown("#### Word Usage Analysis")
                    word_table = gr.HTML()
                    gr.Markdown("#### Word Cloud")
                    word_cloud = gr.HTML()

        with gr.Tab("Interview Mode"):
            gr.Markdown("## Interview Mode – Live Experience")
            gr.Markdown("Answer 5 sequential questions with 90 seconds each.")

            interview_question = gr.Markdown("<div class='question-box'>Press Start to begin your interview</div>")
            interview_video = gr.Video(elem_classes="video-interview")
            interview_timer = gr.Textbox(value="Time Left: 1:30", interactive=False, label="⏱ Timer")
            with gr.Row():
                start_btn = gr.Button("Start Interview", elem_classes="green-button")
                next_btn = gr.Button("Next Question", elem_classes="blue-button", visible=False)

            # Timer component to update every second
            timer = gr.Timer(value=1, active=False)

            # Start interview
            start_btn.click(
                fn=start_interview,
                inputs=[interview_index, time_left, timer_active],
                outputs=[
                    interview_question,
                    interview_index,
                    interview_timer,
                    next_btn,
                    time_left,
                    timer_active,
                    timer_state
                ]
            )

            # Update timer every second if timer_state is True
            def update_timer_if_active(
                index, time_left, timer_active, timer_state,
                interview_question, interview_timer, next_btn
            ):
                if not timer_state:
                    return (
                        interview_question,
                        index,
                        interview_timer,
                        next_btn,
                        time_left,
                        timer_active,
                        timer_state,
                        gr.Timer(active=False)
                    )
                result = run_interview(index, time_left, timer_active)
                return (
                    result[0],  # interview_question
                    result[1],  # index
                    result[2],  # interview_timer
                    result[3],  # next_btn
                    result[4],  # time_left
                    result[5],  # timer_active
                    result[6],  # timer_state
                    gr.Timer(active=True)
                )

            timer.tick(
                fn=update_timer_if_active,
                inputs=[
                    interview_index, time_left, timer_active, timer_state,
                    interview_question, interview_timer, next_btn
                ],
                outputs=[
                    interview_question,
                    interview_index,
                    interview_timer,
                    next_btn,
                    time_left,
                    timer_active,
                    timer_state,
                    timer
                ]
            )

            # Skip to next question
            next_btn.click(
                fn=lambda idx, tl, ta: run_interview(idx, tl, ta, skip=True),
                inputs=[interview_index, time_left, timer_active],
                outputs=[
                    interview_question,
                    interview_index,
                    interview_timer,
                    next_btn,
                    time_left,
                    timer_active,
                    timer_state
                ]
            )

    refresh_btn.click(
        fn=lambda: (f"<div class='question-box'><strong>Your Question:</strong><br>{get_random_question()}</div>", get_random_question()),
        inputs=[],
        outputs=[question_display, question_state]
    )

    reset_btn.click(fn=reset_ui, inputs=[], outputs=[
        question_display, video_input,
        score_box, rank_box, score_table, insights, word_table, word_cloud
    ])

    submit_btn.click(
        fn=save_video,
        inputs=[video_input, question_state],
        outputs=[
            score_box, rank_box, report_plot,
            score_table, insights, word_table, word_cloud
        ]
    )

demo.launch()