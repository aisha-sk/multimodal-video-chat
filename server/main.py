import openai
import numpy as np
from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import datetime
import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
current_video_id = None
stored_embeddings = []

# Helpers
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

# Routes
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

        top = sorted(scored_chunks, key=lambda x: x["score"], reverse=True)[0]

        return jsonify({
            "answer": top["text"],
            "timestamp": top["start_time"],
            "link": top.get("link"),
            "score": top["score"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Server start
if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)
