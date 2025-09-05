# Audio RAG Pipeline - Never Forget a Meeting

An AI-powered audio intelligence system that transforms business meetings into actionable insights. Built on Modal's serverless infrastructure, this pipeline transcribes audio with speaker identification, extracts comprehensive business intelligence, and provides semantic search capabilities through your entire meeting history.

## Key Features

- **Speaker-Aware Transcription**: Powered by AssemblyAI with automatic speaker diarization
- **AI Meeting Intelligence**: Comprehensive extraction of business context, sales opportunities, and action items
- **Semantic Search**: Vector-based search through all your transcripts using OpenAI embeddings
- **RAG Q&A**: Ask questions about your meetings and get AI-powered answers with context
- **Business Analytics**: Track deals, client relationships, competitors, and action items
- **Serverless Architecture**: Scales automatically on Modal's infrastructure
- **Multiple Interfaces**: Streamlit app, standalone HTML, and REST API

## Tech Stack

### Core Infrastructure
- **[Modal](https://modal.com)**: Serverless compute platform for all processing
- **PostgreSQL with pgvector**: Vector database for semantic search (Neon recommended)

### AI/ML Services
- **AssemblyAI**: Professional transcription with speaker diarization
- **Google Gemini 2.5 Flash**: Meeting summarization and metadata extraction  
- **OpenAI Embeddings**: text-embedding-3-small for semantic search

### Languages & Frameworks
- **Python 3.10**: Core application
- **FastAPI**: REST API endpoints
- **Streamlit**: Interactive web interface
- **Vanilla JavaScript**: Standalone HTML interface

## Architecture

```
Audio File Upload
       ↓
AssemblyAI Transcription (with speakers)
       ↓
Gemini Summarization
       ↓
Gemini Metadata Extraction (business intelligence)
       ↓
OpenAI Embeddings Generation
       ↓
PostgreSQL + pgvector Storage
       ↓
Semantic Search & RAG Interface
```

## Quick Start

### Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **PostgreSQL Database**: With pgvector extension (Neon.tech recommended)
3. **API Keys**:
   - AssemblyAI API key
   - OpenAI API key
   - Google Gemini API key

### Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd modal-never-forget
```

2. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your actual API keys and database URL
```

3. **Install Modal CLI**
```bash
pip install modal
modal token new
```

4. **Configure Modal secrets**
```bash
# Create secrets in Modal
modal secret create DATABASE_URL --from-env DATABASE_URL
modal secret create API_KEYS --from-env GEMINI_API_KEY,OPENAI_API_KEY,ASSEMBLYAI_API_KEY
modal secret create API_AUTH_KEY --from-env API_AUTH_KEY
```

5. **Deploy to Modal**
```bash
modal deploy main.py
```

6. **Initialize the database**
```bash
modal run main.py::setup_database
```

## API Endpoints

Once deployed, your endpoints will be available at:

- **Upload Audio**: `POST https://{workspace}--upload-audio.modal.run`
- **Search**: `GET https://{workspace}--search.modal.run?q={query}`
- **Ask Questions**: `GET https://{workspace}--ask.modal.run?q={question}`
- **Business Intelligence**: `GET https://{workspace}--business-intelligence.modal.run?action={action}`
- **Health Check**: `GET https://{workspace}--health.modal.run`

All endpoints require authentication via `Authorization: Bearer {API_AUTH_KEY}` header.

## Frontend Interfaces

### 1. Streamlit App
Full-featured interface with all capabilities:

```bash
pip install streamlit
export API_AUTH_KEY=your-api-key
streamlit run streamlit_ui_enhanced.py
```

Features:
- Audio file upload with progress tracking
- Meeting intelligence dashboard
- Sales pipeline management
- Client relationship tracking
- Competitive analysis
- Action items management

### 2. Standalone HTML
Single-file interface that works in any browser:

1. Open `audio_rag_enhanced_ui.html` in your browser
2. Enter your Modal URL and API key
3. Upload and search audio files

No installation required - perfect for sharing with team members.

## Usage Examples

### Upload and Process Audio
```python
import requests

url = "https://your-workspace--upload-audio.modal.run"
headers = {"Authorization": "Bearer your-api-key"}
files = {"audio_file": open("meeting.mp3", "rb")}
data = {"authorization": "Bearer your-api-key"}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Transcription: {result['transcription']}")
print(f"Summary: {result['summary']}")
print(f"Meeting Intelligence: {result['metadata']}")
```

### Search Transcripts
```python
params = {"q": "budget discussion", "type": "transcription", "limit": 5}
response = requests.get(
    "https://your-workspace--search.modal.run",
    headers=headers,
    params=params
)
```

### Ask Questions (RAG)
```python
params = {"q": "What are the main client objections?", "context_limit": 3}
response = requests.get(
    "https://your-workspace--ask.modal.run",
    headers=headers,
    params=params
)
```

## Meeting Intelligence Extraction

The system automatically extracts:

### Meeting Context
- Meeting type, stage, sentiment, outcome
- Participant details and decision makers
- Duration and language

### Business Intelligence
- Client problems and needs
- Budget and timeline
- Success criteria
- Technical requirements

### Sales Intelligence  
- Deal stage and probability
- Buying signals and objections
- Champions and blockers
- Competitive positioning

### Action Items
- Tasks for consultant and client teams
- Deadlines and milestones
- Next steps and follow-ups

## Local Development

For local testing and debugging:

```bash
# Install dependencies
pip install -r requirements.txt

# Load environment variables
source .env

# Run debug script
python debug_local.py
```

## File Structure

```
modal-never-forget/
├── main.py                    # Core Modal application
├── streamlit_ui_enhanced.py   # Streamlit web interface
├── audio_rag_enhanced_ui.html # Standalone HTML interface
├── debug_local.py            # Local debugging utilities
├── .env.example              # Environment variables template
├── .gitignore               # Git ignore rules
├── .dockerignore            # Docker ignore rules
├── CLAUDE.md                # AI assistant instructions
└── README.md                # This file
```

## Security

- All API endpoints require authentication
- Credentials stored in environment variables
- `.env` file excluded from version control
- Modal secrets for production deployment
- No hardcoded keys in frontend applications

## Supported Audio Formats

- WAV, MP3, MP4, M4A
- FLAC, OGG, WebM  
- Any format supported by FFmpeg
- Maximum file size: 50MB
- Processing timeout: 10 minutes

## Limitations

- 50MB maximum file size per upload
- 10-minute processing timeout
- English language optimized (auto-detection available)
- Vector dimensions: 1536 (OpenAI text-embedding-3-small)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built on [Modal](https://modal.com) serverless platform
- Transcription by [AssemblyAI](https://assemblyai.com)
- AI models by [Google](https://ai.google.dev) and [OpenAI](https://openai.com)
- Vector database by [Neon](https://neon.tech) with pgvector

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the [Modal documentation](https://modal.com/docs)
- Review CLAUDE.md for development guidance

---

Transform your meetings into insights. Never forget what was discussed.