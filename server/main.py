import openai
import numpy as np
import os
import datetime
import subprocess
import glob
import cv2
from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
clip_model = SentenceTransformer("clip-ViT-B-32")


load_dotenv()

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
current_video_id = None
stored_embeddings = []
frame_embeddings = []

# --- HELPERS ---

def extract_video_id(youtube_url):
    query = urlparse(youtube_url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    elif query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            return parse_qs(query.query)['v'][0]
        elif query.path.startswith('/embed/'):
            return query.path.split('/')[2]
        elif query.path.startswith('/v/'):
            return query.path.split('/')[2]
    return None

def format_timestamp(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def chunk_transcript_for_embeddings(transcript, max_words=100):
    chunks = []
    current_chunk = []
    current_word_count = 0
    start_time = 0

    for entry in transcript:
        text = entry["text"]
        word_count = len(text.split())

        if current_word_count + word_count > max_words:
            combined_text = " ".join([e["text"] for e in current_chunk])
            chunks.append({
                "text": combined_text,
                "start": entry["start"],
                "start_time": format_timestamp(start_time),
                "link": f"https://www.youtube.com/watch?v={current_video_id}&t={int(start_time)}s"
            })
            current_chunk = []
            current_word_count = 0
            start_time = entry["start"]

        current_chunk.append(entry)
        current_word_count += word_count

    if current_chunk:
        combined_text = " ".join([e["text"] for e in current_chunk])
        chunks.append({
            "text": combined_text,
            "start": current_chunk[0]["start"],
            "start_time": format_timestamp(current_chunk[0]["start"]),
            "link": f"https://www.youtube.com/watch?v={current_video_id}&t={int(current_chunk[0]['start'])}s"
        })

    return chunks

# --- ROUTES ---

@app.route("/api/embed", methods=["POST"])
def embed_chunks():
    global stored_embeddings, current_video_id
    data = request.json
    url = data.get("url")

    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        current_video_id = video_id
        chunks = chunk_transcript_for_embeddings(transcript)

        enriched = []
        for chunk in chunks:
            text = chunk["text"]
            response = client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            vector = response.data[0].embedding
            enriched.append({**chunk, "embedding": vector})

        stored_embeddings = enriched
        return jsonify({"status": "embeddings stored", "chunks": len(stored_embeddings)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ask", methods=["POST"])
def ask_video():
    global stored_embeddings
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "Missing question"}), 400
    if not stored_embeddings:
        return jsonify({"error": "No embeddings stored"}), 400

    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=question
        )
        question_embedding = response.data[0].embedding

        def cosine_similarity(vec1, vec2):
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        scored_chunks = [
            {
                **chunk,
                "score": cosine_similarity(question_embedding, chunk["embedding"])
            }
            for chunk in stored_embeddings
        ]

        top = sorted(scored_chunks, key=lambda x: x["score"], reverse=True)[:3]

        return jsonify({
            "answer": top[0]["text"],
            "top_chunks": [
                {
                    "link": chunk["link"],
                    "timestamp": chunk["start_time"],
                    "score": chunk["score"]
                }
                for chunk in top
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/summary", methods=["POST"])
def summarize_video_v2():
    global current_video_id
    data = request.json
    url = data.get("url")

    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        current_video_id = video_id

        transcript_lines = []
        for entry in transcript:
            timestamp = format_timestamp(entry["start"])
            text = entry["text"]
            transcript_lines.append(f"[{timestamp}] {text}")

        full_transcript = "\n".join(transcript_lines)

        video_length_secs = transcript[-1]["start"] if transcript else 0
        num_sections = 3
        if video_length_secs > 600:
            num_sections = 5
        if video_length_secs > 1200:
            num_sections = 7

        summary_prompt = f"""
You are an expert video summarizer.

Break the following video transcript into {num_sections} natural sections based on content shifts.

For each section, provide:
1. Section title
2. Start timestamp (hh:mm:ss)
3. 2-3 sentence summary

Use ONLY the timestamps visible in the transcript.

Respond EXACTLY in this JSON format:
[
  {{
    "section": 1,
    "title": "...",
    "start_time": "...",
    "summary": "..."
  }},
  ...
]

Transcript:
{full_transcript}
"""

        import json

        for attempt in range(3):
            chat_response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system", "content": "You are an expert video summarizer."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )

            sections_json = chat_response.choices[0].message.content.strip()

            try:
                sections = json.loads(sections_json)

                enriched_sections = []
                for section in sections:
                    t_parts = section["start_time"].split(":")
                    secs = int(t_parts[0]) * 3600 + int(t_parts[1]) * 60 + int(t_parts[2])

                    enriched_sections.append({
                        "section": section["section"],
                        "title": section["title"],
                        "start_time": section["start_time"],
                        "summary": section["summary"],
                        "link": f"https://www.youtube.com/watch?v={video_id}&t={secs}s"
                    })

                return jsonify(enriched_sections)

            except Exception as e:
                print(f"[summary v2] parse failed (attempt {attempt+1}), retrying... error={e}")

        return jsonify([{"error": "Failed to parse GPT output after retries."}])

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/extract_frames", methods=["POST"])
def extract_frames():
    data = request.json
    url = data.get("url")
    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    try:
        os.makedirs(f"frames/{video_id}", exist_ok=True)
        video_path = f"frames/{video_id}/{video_id}.mp4"
        cmd_download = f"yt-dlp -f mp4 -o '{video_path}' '{url}'"
        subprocess.run(cmd_download, shell=True, check=True)

        frame_output_pattern = f"frames/{video_id}/frame_%04d.jpg"
        cmd_frames = f"ffmpeg -i '{video_path}' -vf fps=1/5 '{frame_output_pattern}'"
        subprocess.run(cmd_frames, shell=True, check=True)

        frame_count = len(glob.glob(f"{frame_output_pattern.replace('%04d', '*')}"))

        return jsonify({"status": "frames extracted", "frame_folder": f"frames/{video_id}", "frames": frame_count})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/embed_frames", methods=["POST"])
def embed_frames():
    global frame_embeddings
    data = request.json
    url = data.get("url")
    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    try:
        frame_folder = f"frames/{video_id}"
        frame_paths = sorted(glob.glob(f"{frame_folder}/frame_*.jpg"))

        frame_embeddings = []
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            image_embedding = clip_model.encode(image)
            frame_embeddings.append({"frame_path": frame_path, "embedding": image_embedding.tolist()})

        return jsonify({"status": "frame embeddings stored", "frames": len(frame_embeddings)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/frame_search", methods=["POST"])
def frame_search():
    global frame_embeddings
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "Missing query"}), 400
    if not frame_embeddings:
        return jsonify({"error": "No frame embeddings stored"}), 400

    try:
        query_embedding = clip_model.encode(query)

        def cosine_similarity(vec1, vec2):
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        scored_frames = [
            {
                "frame_path": frame["frame_path"],
                "score": cosine_similarity(query_embedding, frame["embedding"])
            }
            for frame in frame_embeddings
        ]

        top = sorted(scored_frames, key=lambda x: x["score"], reverse=True)[:5]

        return jsonify({
            "top_frames": [
                {
                    "frame": frame["frame_path"],
                    "score": frame["score"]
                }
                for frame in top
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- SERVER ---

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)
