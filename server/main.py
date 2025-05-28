from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import datetime

app = Flask(__name__)
current_video_id = None  # Global tracker for video ID used in hyperlinking

# ------------------- HELPERS -------------------

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

def chunk_transcript(transcript, gap_threshold=5):
    sections = []
    current_chunk = []
    last_time = 0

    for entry in transcript:
        start = entry["start"]
        if start - last_time > gap_threshold and current_chunk:
            section_text = " ".join([c["text"] for c in current_chunk])
            sections.append({
                "title": section_text[:60] + "...",
                "start": format_timestamp(current_chunk[0]["start"]),
                "link": f"https://www.youtube.com/watch?v={current_video_id}&t={int(current_chunk[0]['start'])}s"
            })
            current_chunk = []
        current_chunk.append(entry)
        last_time = start

    if current_chunk:
        section_text = " ".join([c["text"] for c in current_chunk])
        sections.append({
            "title": section_text[:60] + "...",
            "start": format_timestamp(current_chunk[0]["start"]),
            "link": f"https://www.youtube.com/watch?v={current_video_id}&t={int(current_chunk[0]['start'])}s"
        })

    return sections

# ------------------- ROUTES -------------------

@app.route("/api/transcript", methods=["POST"])
def get_transcript():
    data = request.json
    video_url = data.get("url")
    video_id = extract_video_id(video_url)

    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return jsonify(transcript)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/sections", methods=["POST"])
def get_sections():
    data = request.json
    video_url = data.get("url")
    global current_video_id
    current_video_id = extract_video_id(video_url)

    if not current_video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    try:
        transcript = YouTubeTranscriptApi.get_transcript(current_video_id)
        sections = chunk_transcript(transcript)
        return jsonify(sections)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------- SERVER -------------------

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)
