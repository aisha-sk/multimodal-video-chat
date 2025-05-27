# multimodal video chat

this is a full-stack project that lets you interact with youtube videos by retrieving and analyzing their transcripts. the goal is to eventually support both text-based and visual queries through a simple web interface.

## about the project

this project is being built as part of headstarter's accelerated program. it follows one of the suggested multimodal tracks and is focused on building a working prototype with both nlp and visual understanding capabilities.

## what’s working

so far, the backend is set up to extract youtube transcripts through an api:

- built a flask backend that accepts a youtube url and returns its transcript
- uses the `youtube-transcript-api` to fetch subtitle data
- parses and validates youtube urls to extract the video id
- endpoint: `post /api/transcript` with body like `{"url": "https://youtube.com/watch?v=..."}`
- tested with real urls and returns structured transcript output (text with timestamps)

## what’s next

- build the frontend interface in react
- allow users to paste a link and view the transcript on screen
- add chat functionality to ask natural language questions based on the transcript
- break transcript into chunks and generate vector embeddings (rag style)
- explore adding visual question answering later using gemini vision or clip model to analyze video frames

## original readme content below (create react app defaults)

this project was bootstrapped with [create react app](https://github.com/facebook/create-react-app).

