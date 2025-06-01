#!/bin/bash

# === CONFIG ===
VIDEO_URL="https://www.youtube.com/watch?v=y8kTYCex8RU&ab_channel=Animalogic"
QUERY_TEXT="gun"

# === TEST SEQUENCE ===

echo "STEP 1️⃣ /api/embed"
curl -X POST http://127.0.0.1:5000/api/embed \
  -H "Content-Type: application/json" \
  -d "{\"url\":\"$VIDEO_URL\"}"
echo -e "\n---------------------\n"

echo "STEP 2️⃣ /api/extract_frames"
curl -X POST http://127.0.0.1:5000/api/extract_frames \
  -H "Content-Type: application/json" \
  -d "{\"url\":\"$VIDEO_URL\"}"
echo -e "\n---------------------\n"

echo "STEP 3️⃣ /api/embed_frames"
curl -X POST http://127.0.0.1:5000/api/embed_frames \
  -H "Content-Type: application/json" \
  -d "{\"url\":\"$VIDEO_URL\"}"
echo -e "\n---------------------\n"

echo "STEP 4️⃣ /api/frame_search"
curl -X POST http://127.0.0.1:5000/api/frame_search \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"$QUERY_TEXT\"}"
echo -e "\n---------------------\n"

echo "✅ TEST COMPLETE!"
