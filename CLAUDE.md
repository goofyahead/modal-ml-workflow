# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Audio RAG (Retrieval-Augmented Generation) Pipeline** built on Modal Labs cloud platform. The application transcribes audio files using OpenAI Whisper, generates summaries with Google's Gemini, creates vector embeddings, and stores everything in a PostgreSQL database with pgvector for semantic search.

## Core Architecture

- **Single File Application**: `main.py` contains the entire pipeline
- **Modal Functions**: Serverless functions running on Modal's infrastructure
- **Database**: PostgreSQL with pgvector extension for vector similarity search
- **AI Models**: 
  - Whisper (base model) for transcription (GPU)
  - Gemini 1.5 Flash for summarization (CPU)
  - OpenAI text-embedding-3-small for embeddings (CPU)

## Essential Commands

### Deploy and Run
```bash
# Deploy the entire application to Modal
modal deploy main.py

# Initialize database (run once after deployment)
modal run main.py::setup_database

# Run functions locally for development
modal run main.py::transcribe_audio --audio-bytes <bytes> --filename "test.wav"
modal run main.py::semantic_search_v2 --query "your search query"
```

### Live Deployment URLs
- **Upload Audio**: https://goofyahead--upload-audio.modal.run
- **Search**: https://goofyahead--search.modal.run
- **Ask Questions**: https://goofyahead--ask.modal.run
- **Business Intelligence**: https://goofyahead--business-intelligence.modal.run
- **Health Check**: https://goofyahead--health.modal.run
- **Setup Database**: https://goofyahead--setup.modal.run

### Development and Testing
```bash
# Run in development mode with hot reloading
modal serve main.py

# Test individual functions
modal run main.py::<function_name>
```

## Required Secrets

The application requires these Modal secrets to be configured:
- `DATABASE_URL`: PostgreSQL connection string (Neon or similar)
- `API_KEYS`: Combined secret containing GEMINI_API_KEY and OPENAI_API_KEY
- `API_AUTH_KEY`: API key for endpoint authentication (protects all web endpoints)

Configure secrets with: `modal secret create <name>`

## Security

All web endpoints require API key authentication via `Authorization: Bearer <your-api-key>` header. The API key must match the `API_AUTH_KEY` secret configured in Modal.

## Key Functions and Pipeline

1. **Audio Processing Pipeline** (`process_audio_pipeline`): 
   - Transcribes audio → Summarizes → Generates embeddings → Stores in DB

2. **Database Setup** (`setup_database`):
   - Creates `audio_transcripts` table with vector columns
   - Sets up HNSW indexes for fast similarity search

3. **Search Functions**:
   - `semantic_search`: Vector similarity search
   - `rag_answer`: Question answering using retrieved context

4. **Web Endpoints** (using `@modal.fastapi_endpoint`):
   - `/upload-audio` (POST): Upload and process audio files
   - `/search` (GET): Semantic search through transcripts  
   - `/ask` (GET): RAG-based question answering
   - `/setup` (POST): Initialize database

## Database Schema

The `audio_transcripts` table includes:
- Transcription and summary text
- Vector embeddings (1536 dimensions)
- Full-text search capabilities
- Metadata and tags support

## User Interfaces

### Streamlit Web App (`streamlit_ui.py`)
```bash
# Install and run Streamlit interface
pip install streamlit requests
export API_AUTH_KEY="your-api-key"
streamlit run streamlit_ui.py
```

### Standalone HTML Interface (`audio_rag_ui.html`)
- **Single file**: Self-contained HTML with JavaScript
- **No dependencies**: Just open in any web browser
- **Shareable**: Send the HTML file to anyone
- **Usage**: Configure Modal URL and API key in the interface

Both interfaces provide:
- File upload with drag-and-drop for audio files
- Real-time transcription and summarization  
- Semantic search through all transcripts
- RAG-based question answering interface

## Architecture Notes

- Functions are containerized with specific resource allocations (CPU/GPU)
- Transcription uses GPU (T4) for Whisper **small** model (10-minute timeout)
- Enhanced metadata extraction uses Gemini 2.5 Flash Lite for business intelligence
- Vector similarity uses cosine distance with HNSW indexes
- 50MB file size limit for audio uploads
- **Audio formats**: Supports any format FFmpeg can decode (including OGG)
- **NEW**: Enhanced with mock speaker diarization and comprehensive business intelligence

## Business Intelligence Features

The system now includes comprehensive meeting intelligence extraction:

### Structured Metadata Fields
- **Meeting Context**: Type, stage, sentiment, outcome
- **Participants**: Consultant team, client team, decision makers
- **Technical Details**: Current tech stack, challenges, solutions
- **Business Intelligence**: Problems, needs, budget, timeline
- **Sales Intelligence**: Deal stage, probability, buying signals, objections
- **Competitive Landscape**: Competitors, advantages, concerns
- **Action Items**: Tasks, milestones, next steps
- **Strategic Insights**: Takeaways, recommendations, opportunities

### API Usage
Use the consolidated business intelligence endpoint with action parameters:
```bash
# Get deals pipeline
curl "https://goofyahead--business-intelligence.modal.run?action=deals&authorization=Bearer YOUR_API_KEY"

# Get high probability deals
curl "https://goofyahead--business-intelligence.modal.run?action=high-probability-deals&authorization=Bearer YOUR_API_KEY"

# Get client insights
curl "https://goofyahead--business-intelligence.modal.run?action=client-insights&client=CompanyName&authorization=Bearer YOUR_API_KEY"

# Other actions: competitors, action-items, all-meetings
```