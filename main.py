import modal
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App("audio-rag-pipeline")

# Image with AssemblyAI for transcription and speaker diarization
image = modal.Image.debian_slim(python_version="3.10").run_commands([
    "pip install assemblyai google-generativeai openai asyncpg fastapi python-multipart",
    "python -c 'import assemblyai; print(f\"AssemblyAI SDK installed successfully\")'"
])

# Import FastAPI components after image definition
try:
    from fastapi import Form, File, Response, UploadFile, Header
except ImportError:
    # Will be available in Modal runtime - create dummy functions that return None
    def File(*args, **kwargs): return None
    def Form(*args, **kwargs): return None
    def Header(*args, **kwargs): return None
    def UploadFile(*args, **kwargs): return None
    def Response(*args, **kwargs): return None

# Secrets for database and APIs
SECRETS = [
    modal.Secret.from_name("DATABASE_URL"),  # Your Neon connection string
    modal.Secret.from_name("API_KEYS"), # External services keys
    modal.Secret.from_name("API_AUTH_KEY"),  # For endpoint authentication
]


# Initialize database with pgvector
def validate_api_key(authorization: str):
    """Validate API key from Authorization header"""
    from fastapi import HTTPException
    import os
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    
    token = authorization.replace("Bearer ", "")
    expected_key = os.environ.get("API_AUTH_KEY")
    
    if not expected_key or token != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.function(
    image=image,
    cpu=1.0,
    secrets=SECRETS
)
async def setup_database():
    """Initialize database with pgvector and create tables"""
    import asyncpg
    import os

    conn = await asyncpg.connect(os.environ["DATABASE_URL"])

    try:
        # Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create new v2 table with enhanced metadata structure
        await conn.execute("""
                           CREATE TABLE IF NOT EXISTS audio_transcripts
                           (
                               id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                               filename TEXT NOT NULL,
                               transcription TEXT NOT NULL,
                               summary TEXT,
                               segments JSONB,
                               language TEXT DEFAULT 'unknown',
                               created_at TIMESTAMP DEFAULT NOW(),
                               updated_at TIMESTAMP DEFAULT NOW(),

                               -- Vector embeddings (OpenAI text-embedding-3-small = 1536 dimensions)
                               transcription_embedding VECTOR(1536),
                               summary_embedding VECTOR(1536),

                               -- Enhanced structured metadata
                               meeting_metadata JSONB DEFAULT '{}'::jsonb,
                               tags TEXT[] DEFAULT '{}',

                               -- Full-text search
                               transcription_search tsvector GENERATED ALWAYS AS (
                                   to_tsvector('english', transcription || ' ' || COALESCE(summary, ''))
                               ) STORED
                           )
                           """)

        # Create indexes for vector similarity search on v2 table
        await conn.execute("""
                           CREATE INDEX IF NOT EXISTS idx_transcription_embedding_hnsw
                               ON audio_transcripts USING hnsw (transcription_embedding vector_cosine_ops);
                           """)

        await conn.execute("""
                           CREATE INDEX IF NOT EXISTS idx_summary_embedding_hnsw
                               ON audio_transcripts USING hnsw (summary_embedding vector_cosine_ops);
                           """)

        # Full-text search index on v2 table
        await conn.execute("""
                           CREATE INDEX IF NOT EXISTS idx_transcription_search
                               ON audio_transcripts USING gin(transcription_search);
                           """)
                           
        # JSONB index for meeting metadata
        await conn.execute("""
                           CREATE INDEX IF NOT EXISTS idx_meeting_metadata
                               ON audio_transcripts USING gin(meeting_metadata);
                           """)
                           
        # Index on segments for speaker queries
        await conn.execute("""
                           CREATE INDEX IF NOT EXISTS idx_segments
                               ON audio_transcripts USING gin(segments);
                           """)

        print("Database initialized with pgvector support!")
        return {"status": "success"}

    finally:
        await conn.close()


# Generate embeddings
@app.function(
    image=image,
    cpu=1.0,
    secrets=SECRETS
)
def generate_embeddings(text: str) -> List[float]:
    """Generate embeddings using OpenAI"""
    from openai import OpenAI
    import os

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format="float"
    )

    return response.data[0].embedding


# AssemblyAI transcription with speaker diarization (CPU)
@app.function(
    image=image,
    cpu=2.0,
    timeout=600,
    secrets=SECRETS
)
def transcribe_audio(audio_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Transcribe audio using AssemblyAI with speaker diarization"""
    import assemblyai as aai
    import tempfile
    import os
    import time

    # Get AssemblyAI API key
    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise ValueError("ASSEMBLYAI_API_KEY is required for transcription")
    
    aai.settings.api_key = api_key
    
    # Save audio to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

    try:
        logger.info("🎤 Starting AssemblyAI transcription with speaker diarization")
        
        # Configure transcription with speaker diarization
        config = aai.TranscriptionConfig(
            speaker_labels=True,  # Enable speaker diarization
            auto_highlights=False,
            language_detection=True  # Auto-detect language
        )
        
        # Create transcriber and submit job
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(tmp_path)
        
        # Wait for transcription to complete
        while transcript.status not in [aai.TranscriptStatus.completed, aai.TranscriptStatus.error]:
            logger.info("⏳ Waiting for transcription to complete...")
            time.sleep(2)
            transcript = transcriber.get_transcript(transcript.id)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"AssemblyAI transcription failed: {transcript.error}")
        
        logger.info("✅ AssemblyAI transcription completed successfully")
        
        # Format the output with speaker identification
        full_transcription = ""
        segments_with_speakers = []
        
        if transcript.utterances:
            # Use speaker-labeled utterances
            for utterance in transcript.utterances:
                speaker = f"Speaker {utterance.speaker}"
                text = utterance.text.strip()
                start_time = utterance.start / 1000.0  # Convert ms to seconds
                end_time = utterance.end / 1000.0
                
                if text:  # Only add non-empty segments
                    full_transcription += f"[{speaker}] {text}\n"
                    segments_with_speakers.append({
                        "speaker": speaker,
                        "text": text,
                        "start": start_time,
                        "end": end_time
                    })
        else:
            # Fallback to regular segments without speaker labels
            logger.warning("⚠️ No speaker diarization available, using regular segments")
            for sentence in transcript.sentences:
                text = sentence.text.strip()
                start_time = sentence.start / 1000.0
                end_time = sentence.end / 1000.0
                
                if text:
                    full_transcription += f"[Unknown] {text}\n"
                    segments_with_speakers.append({
                        "speaker": "Unknown",
                        "text": text,
                        "start": start_time,
                        "end": end_time
                    })
        
        # Get detected language if available
        language = getattr(transcript, 'language_code', 'en') or 'en'
        
        return {
            "status": "success",
            "transcription": full_transcription.strip(),
            "segments": segments_with_speakers,
            "language": language,
            "filename": filename,
            "confidence": getattr(transcript, 'confidence', 0.95)
        }
        
    except Exception as e:
        logger.error("❌ ASSEMBLYAI TRANSCRIPTION FAILED: %s", str(e))
        return {
            "status": "error",
            "transcription": f"Error during transcription: {str(e)}",
            "segments": [],
            "language": "unknown",
            "filename": filename,
            "error": str(e)
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# Gemini summarization (CPU)
@app.function(
    image=image,
    cpu=2.0,
    timeout=60,
    secrets=SECRETS
)
def summarize_text(transcription_data: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize using Gemini Flash"""
    import google.generativeai as genai
    import os

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel('models/gemini-2.5-flash-lite')

    prompt = f"""
    Provide a concise summary of this audio transcription:
    "{transcription_data['transcription']}"

    Focus on:
    - Key topics and main points
    - Action items or decisions
    - Important names, dates, or numbers
    - 2-3 sentences maximum
    """

    try:
        response = model.generate_content(prompt)
        return {
            **transcription_data,
            "summary": response.text,
            "processed_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            **transcription_data,
            "summary": f"Error: {str(e)}",
            "processed_at": datetime.utcnow().isoformat()
        }


# Metadata extraction (CPU)
@app.function(
    image=image,
    cpu=2.0,
    timeout=120,
    secrets=SECRETS
)
def extract_metadata(transcription_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured metadata from transcription using Gemini"""
    import google.generativeai as genai
    import os
    import json

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel('models/gemini-2.5-flash-lite')

    prompt = f"""
    You are analyzing a business meeting transcription between a CONSULTANT company (service provider) and a CLIENT company (potential buyer).
    The speakers labeled as consultants are selling/offering services. The client is evaluating whether to purchase these services.
    
    Meeting Transcription:
    "{transcription_data['transcription']}"

    Extract comprehensive metadata as JSON. Analyze the conversation dynamics, business context, and sales opportunities.

    Return ONLY valid JSON with these fields:
    {{
        "meeting_context": {{
            "meeting_type": "discovery_call|technical_review|proposal_presentation|follow_up|negotiation|other",
            "meeting_stage": "initial_contact|requirements_gathering|solution_design|negotiation|closing|post_sale",
            "meeting_date": "YYYY-MM-DD if mentioned",
            "meeting_duration": "minutes if determinable",
            "meeting_sentiment": "very_positive|positive|neutral|concerned|resistant",
            "meeting_outcome": "brief description of how meeting ended"
        }},
        
        "participants": {{
            "consultant_team": [
                {{"name": "Person Name", "role": "Title/Role", "company": "Consultant Company"}}
            ],
            "client_team": [
                {{"name": "Person Name", "role": "Title/Role", "is_decision_maker": true/false}}
            ],
            "client_company_name": "Client Company Name",
            "client_company_details": {{
                "industry": "Industry sector",
                "company_size": "startup|small|medium|enterprise",
                "maturity_level": "early_stage|growth|established|enterprise"
            }}
        }},
        
        "technical_details": {{
            "client_current_tech_stack": ["technology1", "technology2"],
            "client_technical_challenges": ["challenge1", "challenge2"],
            "proposed_technical_solutions": ["solution1", "solution2"],
            "integration_requirements": ["requirement1", "requirement2"],
            "technical_constraints": ["constraint1", "constraint2"]
        }},
        
        "business_intelligence": {{
            "client_problems": [
                {{"problem": "description", "severity": "low|medium|high|critical", "impact": "business impact"}}
            ],
            "client_needs": ["need1", "need2"],
            "client_goals": ["goal1", "goal2"],
            "success_criteria": ["criteria1", "criteria2"],
            "budget_mentioned": {{"discussed": true/false, "amount": "amount or range if mentioned", "budget_authority": "person if mentioned"}},
            "timeline_urgency": "immediate|short_term|medium_term|long_term|no_urgency",
            "decision_timeline": "when decision will be made if mentioned"
        }},
        
        "consultant_offerings": {{
            "services_discussed": ["service1", "service2"],
            "unique_value_propositions": ["value_prop1", "value_prop2"],
            "case_studies_mentioned": ["case1", "case2"],
            "pricing_discussed": {{"mentioned": true/false, "details": "pricing details if any"}},
            "consultant_strengths_highlighted": ["strength1", "strength2"]
        }},
        
        "competitive_landscape": {{
            "competitors_mentioned": ["competitor1", "competitor2"],
            "competitive_advantages": ["advantage1", "advantage2"],
            "competitive_concerns": ["concern1", "concern2"],
            "why_considering_us": ["reason1", "reason2"]
        }},
        
        "sales_intelligence": {{
            "buying_signals": ["signal1", "signal2"],
            "objections_raised": ["objection1", "objection2"],
            "concerns_expressed": ["concern1", "concern2"],
            "risk_factors": ["risk1", "risk2"],
            "champions_identified": ["person who seems supportive"],
            "blockers_identified": ["person or issue blocking progress"],
            "estimated_deal_value": "estimated value if discussable",
            "probability_to_close": "0-100 percentage based on conversation",
            "deal_stage": "prospecting|qualification|proposal|negotiation|closing|closed_won|closed_lost"
        }},
        
        "action_items": {{
            "consultant_action_items": [
                {{"task": "description", "owner": "person", "deadline": "date if mentioned"}}
            ],
            "client_action_items": [
                {{"task": "description", "owner": "person", "deadline": "date if mentioned"}}
            ],
            "key_milestones": [
                {{"date": "YYYY-MM-DD or description", "deliverable": "what is needed", "responsible_party": "consultant|client"}}
            ],
            "next_steps": ["step1", "step2"],
            "follow_up_date": "YYYY-MM-DD if scheduled"
        }},
        
        "insights": {{
            "key_takeaways": ["takeaway1", "takeaway2"],
            "recommended_actions": ["action1", "action2"],
            "potential_upsell_opportunities": ["opportunity1", "opportunity2"],
            "relationship_building_notes": ["note1", "note2"],
            "red_flags": ["flag1", "flag2"],
            "green_flags": ["flag1", "flag2"]
        }}
    }}

    Analysis Guidelines:
    - Return ONLY the JSON object, no other text
    - Use null for missing information, empty arrays [] for no items
    - Be specific and extract actual names, companies, and details from the conversation
    - Infer sentiment and buying probability from conversation tone and content
    - Identify decision makers by titles like CEO, CTO, VP, Director, Manager
    - Distinguish between consultant (seller) and client (buyer) based on context
    - Pay attention to verbal cues indicating interest, concern, or commitment
    - Note any mentions of timeline, budget, or decision process
    - Capture competitive intelligence and differentiation points
    - Identify both technical and business value propositions
    """

    try:
        response = model.generate_content(prompt)
        
        # Clean the response text and try to parse JSON
        response_text = response.text.strip()
        
        # Remove any markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
            
        response_text = response_text.strip()
        
        # Parse the JSON
        metadata = json.loads(response_text)
        
        return {
            **transcription_data,
            "metadata": metadata,
            "metadata_extracted_at": datetime.utcnow().isoformat()
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse metadata JSON: {e}")
        logger.error(f"Raw response: {response.text}")
        return {
            **transcription_data,
            "metadata": {
                "error": f"JSON parsing error: {str(e)}",
                "raw_response": response.text
            },
            "metadata_extracted_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Metadata extraction error: {e}")
        return {
            **transcription_data,
            "metadata": {
                "error": f"Extraction error: {str(e)}"
            },
            "metadata_extracted_at": datetime.utcnow().isoformat()
        }


# Store with vector embeddings in v2 table
@app.function(
    image=image,
    cpu=1.0,
    secrets=SECRETS
)
async def store_with_embeddings(result_data: Dict[str, Any]) -> str:
    """Store transcript with vector embeddings in v2 table"""
    import asyncpg
    import os

    # Generate embeddings (wait for remote function calls to complete)
    transcription_embedding = await generate_embeddings.remote.aio(result_data["transcription"])
    summary_embedding = await generate_embeddings.remote.aio(result_data.get("summary", ""))
    
    # Convert embeddings to string format for PostgreSQL vector type
    transcription_vector_str = f"[{','.join(map(str, transcription_embedding))}]"
    summary_vector_str = f"[{','.join(map(str, summary_embedding))}]"

    conn = await asyncpg.connect(os.environ["DATABASE_URL"])

    try:
        record_id = await conn.fetchval("""
                                        INSERT INTO audio_transcripts
                                        (filename, transcription, summary, segments, language, transcription_embedding,
                                         summary_embedding, meeting_metadata)
                                        VALUES ($1, $2, $3, $4, $5, $6::vector, $7::vector, $8) RETURNING id
                                        """,
                                        result_data["filename"],
                                        result_data["transcription"],
                                        result_data.get("summary", ""),
                                        json.dumps(result_data.get("segments", [])),  # Speaker segments
                                        result_data.get("language", "unknown"),
                                        transcription_vector_str,  # Vector embedding as string
                                        summary_vector_str,  # Vector embedding as string
                                        json.dumps({
                                            **result_data.get("metadata", {}),  # Structured meeting metadata
                                            "processed_at": result_data.get("processed_at"),
                                            "metadata_extracted_at": result_data.get("metadata_extracted_at"),
                                            "model_version": "assemblyai",
                                            "summary_model": "gemini-2.5-flash-lite"
                                        })
                                        )

        print(f"Stored with embeddings in v2 table, ID: {record_id}")
        return str(record_id)

    finally:
        await conn.close()


# RAW VERSION FOR DEBUGGING
async def _raw_semantic_search(
        query: str,
        search_type: str = "transcription",  # "transcription" or "summary"
        limit: int = 5
) -> List[Dict[str, Any]]:
    """RAW VERSION - Semantic search using vector similarity on v2 table"""
    import asyncpg
    import os

    # For local debugging, we need to generate embeddings differently
    # This would need OpenAI API call directly rather than Modal remote call
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
    except Exception as e:
        print(f"Error generating embeddings locally: {e}")
        # Return empty results if embedding generation fails
        return []
    
    # Convert embedding to string format for PostgreSQL vector type
    query_vector_str = f"[{','.join(map(str, query_embedding))}]"

    conn = await asyncpg.connect(os.environ["DATABASE_URL"])

    try:
        # Choose which embedding column to search
        embedding_column = f"{search_type}_embedding"

        results = await conn.fetch(f"""
            SELECT 
                id, filename, transcription, summary, segments, meeting_metadata, language, created_at,
                1 - (transcription_embedding <=> $1::vector) as transcription_similarity,
                1 - (summary_embedding <=> $1::vector) as summary_similarity
            FROM audio_transcripts 
            WHERE {embedding_column} IS NOT NULL
            ORDER BY {embedding_column} <=> $1::vector
            LIMIT $2
        """, query_vector_str, limit)

        return [
            {
                **dict(record),
                "id": str(record["id"]),
                "created_at": record["created_at"].isoformat(),
                "segments": record["segments"] if record["segments"] else [],
                "meeting_metadata": record["meeting_metadata"] if record["meeting_metadata"] else {}
            }
            for record in results
        ]

    finally:
        await conn.close()

# RAG: Semantic search using vectors (v2 table)
@app.function(
    image=image,
    cpu=1.0,
    secrets=SECRETS
)
async def semantic_search(
        query: str,
        search_type: str = "transcription",  # "transcription" or "summary"
        limit: int = 5
) -> List[Dict[str, Any]]:
    """MODAL WRAPPER - Semantic search using vector similarity on v2 table"""
    import asyncpg
    import os

    # Generate query embedding (wait for remote function call to complete)
    query_embedding = await generate_embeddings.remote.aio(query)
    
    # Convert embedding to string format for PostgreSQL vector type
    query_vector_str = f"[{','.join(map(str, query_embedding))}]"

    conn = await asyncpg.connect(os.environ["DATABASE_URL"])

    try:
        # Choose which embedding column to search
        embedding_column = f"{search_type}_embedding"

        results = await conn.fetch(f"""
            SELECT 
                id, filename, transcription, summary, segments, meeting_metadata, language, created_at,
                1 - (transcription_embedding <=> $1::vector) as transcription_similarity,
                1 - (summary_embedding <=> $1::vector) as summary_similarity
            FROM audio_transcripts 
            WHERE {embedding_column} IS NOT NULL
            ORDER BY {embedding_column} <=> $1::vector
            LIMIT $2
        """, query_vector_str, limit)

        return [
            {
                **dict(record),
                "id": str(record["id"]),
                "created_at": record["created_at"].isoformat(),
                "segments": record["segments"] if record["segments"] else [],
                "meeting_metadata": record["meeting_metadata"] if record["meeting_metadata"] else {}
            }
            for record in results
        ]

    finally:
        await conn.close()


# RAG: Answer questions using retrieved context (v2 table)
@app.function(
    image=image,
    cpu=2.0,
    secrets=SECRETS
)
def rag_answer(query: str, context_limit: int = 3) -> Dict[str, Any]:
    """Answer questions using RAG with retrieved transcripts from v2 table"""
    import google.generativeai as genai
    import os

    # Get relevant transcripts
    search_results = semantic_search.remote(query, "transcription", context_limit)

    if not search_results:
        return {
            "answer": "No relevant transcripts found.",
            "sources": [],
            "query": query
        }

    # Build context from retrieved transcripts
    context_parts = []
    sources = []

    for i, result in enumerate(search_results):
        context_parts.append(f"""
Transcript {i + 1} (File: {result['filename']}):
Transcription: {result['transcription']}
Summary: {result.get('summary', 'No summary')}
Similarity: {result['transcription_similarity']:.3f}
---
        """)
        sources.append({
            "id": result["id"],
            "filename": result["filename"],
            "similarity": result["transcription_similarity"]
        })

    context = "\n".join(context_parts)

    # Generate answer using Gemini
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel('models/gemini-2.5-flash-lite')

    prompt = f"""
Based on the following audio transcripts, answer the user's question.

CONTEXT:
{context}

USER QUESTION: {query}

Instructions:
- Answer based only on the provided transcripts
- If the transcripts don't contain enough information, say so
- Reference specific transcripts when possible
- Be concise but comprehensive
    """

    try:
        response = model.generate_content(prompt)
        return {
            "answer": response.text,
            "sources": sources,
            "query": query,
            "context_used": len(search_results)
        }
    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": sources,
            "query": query
        }


# Enhanced processing pipeline with metadata extraction
@app.function(
    image=image,
    cpu=1.0,
    timeout=900,
    secrets=SECRETS
)
def process_audio_pipeline(audio_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Enhanced pipeline: transcribe → summarize → extract metadata → embed → store"""
    logger.info("🎬 ENHANCED PIPELINE STARTED - Processing '%s' with %d bytes", filename, len(audio_bytes))

    try:
        # Step 1: Transcribe with speaker diarization (GPU)
        logger.info("🎤 STEP 1: Starting transcription with AssemblyAI...")
        transcription_result = transcribe_audio.remote(audio_bytes, filename)
        
        # Check transcription status - STOP pipeline if failed
        if transcription_result.get("status") != "success":
            error_msg = transcription_result.get("error", "Unknown transcription error")
            logger.error("❌ TRANSCRIPTION FAILED: %s", error_msg)
            raise Exception(f"Transcription failed: {error_msg}")
            
        logger.info("✅ Transcription completed successfully")
        logger.info("📝 Transcription preview: %s...", str(transcription_result)[:200])

        # Step 2: Summarize (CPU)
        logger.info("📄 STEP 2: Starting summarization with Gemini...")
        summary_result = summarize_text.remote(transcription_result)
        logger.info("✅ Summarization completed")
        logger.info("📋 Summary preview: %s...", str(summary_result)[:200])

        # Step 3: Extract metadata (CPU)
        logger.info("🏷️  STEP 3: Extracting meeting metadata...")
        metadata_result = extract_metadata.remote(summary_result)
        logger.info("✅ Metadata extraction completed")
        logger.info("🏢 Metadata preview: %s...", str(metadata_result.get("metadata", {}))[:200])

        # Step 4: Generate embeddings and store in v2 table (CPU)
        logger.info("🗄️  STEP 4: Generating embeddings and storing in v2 table...")
        record_id = store_with_embeddings.remote(metadata_result)
        logger.info("✅ Storage completed with record_id: %s", record_id)

        final_result = {
            "record_id": record_id,
            "status": "completed",
            **metadata_result
        }
        
        logger.info("🎊 ENHANCED PIPELINE COMPLETED - Final result keys: %s", list(final_result.keys()))
        return final_result

    except Exception as e:
        logger.error("❌ ENHANCED PIPELINE ERROR: %s", str(e))
        logger.error("🔍 Error type: %s", type(e).__name__)
        import traceback
        logger.error("📚 Pipeline traceback: %s", traceback.format_exc())
        raise e


# FastAPI components will be available in Modal runtime

# Web endpoints  
@app.function(image=image, cpu=1.0, timeout=900, secrets=SECRETS)
@modal.fastapi_endpoint(method="POST", label="upload-audio")  
async def upload_audio_endpoint(
    audio_file: UploadFile = File(...),
    authorization: str = Form(None)
) -> Dict[str, Any]:
    """Upload and process audio file"""
    
    logger.info("🎵 UPLOAD ENDPOINT STARTED - Received request at %s", datetime.utcnow().isoformat())
    
    try:
        logger.info("🔍 Received - audio_file: %s, authorization: %s", 
                   f"'{audio_file.filename}' ({audio_file.content_type})" if audio_file else "None",
                   "Bearer ***" if authorization else "None")
        
        # Read the file contents
        audio_bytes = await audio_file.read()
        filename = audio_file.filename or "uploaded_audio.wav"
        
        logger.info("🔐 Validating API key...")
        logger.info("📋 Authorization received: %s", "Bearer ***" if authorization else "None")
        validate_api_key(authorization)
        logger.info("✅ API key validated successfully")
        
        logger.info("📁 Checking audio file...")
        if not audio_bytes:
            logger.error("❌ No audio file provided")
            return {"error": "No audio file provided"}

        # audio_bytes and filename already set above
        file_size_mb = len(audio_bytes) / 1024 / 1024
        
        logger.info("📊 Audio file details: filename='%s', size=%.2fMB", filename, file_size_mb)

        if len(audio_bytes) > 50 * 1024 * 1024:  # 50MB limit
            logger.error("❌ File too large: %.2fMB > 50MB limit", file_size_mb)
            return {"error": "File too large (max 50MB)"}

        # Process through enhanced pipeline
        logger.info("🚀 Starting enhanced audio processing pipeline...")
        logger.info("📝 Calling process_audio_pipeline.remote() with %d bytes", len(audio_bytes))
        
        result = process_audio_pipeline.remote(audio_bytes, filename)
        
        logger.info("✅ Pipeline completed successfully")
        logger.info("📋 Pipeline result keys: %s", list(result.keys()) if result else 'None')

        response_data = {"status": "success", **result}
        
        logger.info("🎯 Returning success response with %d chars", len(str(response_data)))
        
        return response_data

    except Exception as e:
        logger.error("❌ UPLOAD ENDPOINT ERROR: %s", str(e))
        logger.error("🔍 Error type: %s", type(e).__name__)
        import traceback
        logger.error("📚 Full traceback: %s", traceback.format_exc())
        
        error_data = {"error": str(e), "status": "error"}
        return error_data


@app.function(image=image, cpu=1.0, secrets=SECRETS)
@modal.fastapi_endpoint(method="GET", label="search")
def search_endpoint(
        q: str,  # Search query
        type: str = "transcription",  # "transcription" or "summary"
        limit: int = 5,
        authorization: str = Header(None)
) -> Dict[str, Any]:
    """Semantic search through transcripts"""
    try:
        validate_api_key(authorization)
        
        results = semantic_search.remote(q, type, limit)
        response_data = {
            "status": "success",
            "query": q,
            "search_type": type,
            "results": results
        }
        
        from fastapi import Response
        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )
    except Exception as e:
        from fastapi import Response
        error_data = {"error": str(e), "status": "error"}
        return Response(
            content=json.dumps(error_data),
            media_type="application/json",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )


@app.function(image=image, cpu=2.0, secrets=SECRETS)
@modal.fastapi_endpoint(method="GET", label="ask")
def rag_endpoint(
        q: str,  # Question
        context_limit: int = 3,
        authorization: str = Header(None)
) -> Dict[str, Any]:
    """Ask questions about your transcripts using RAG"""
    try:
        validate_api_key(authorization)
        
        result = rag_answer.remote(q, context_limit)
        response_data = {"status": "success", **result}
        
        from fastapi import Response
        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS", 
                "Access-Control-Allow-Headers": "*"
            }
        )
    except Exception as e:
        from fastapi import Response
        error_data = {"error": str(e), "status": "error"}
        return Response(
            content=json.dumps(error_data),
            media_type="application/json",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )


# Health endpoint
@app.function(image=image, cpu=0.125, secrets=SECRETS)
@modal.fastapi_endpoint(method="GET", label="health")
def health_endpoint() -> Dict[str, Any]:
    """Enhanced health check endpoint with secret verification"""
    import os
    
    def mask_secret(value: str) -> str:
        """Mask secret showing only last 4 characters"""
        if not value:
            return "NOT_SET"
        if len(value) <= 4:
            return "***" + value
        return "***" + value[-4:]
    
    # Check available secrets/environment variables
    secret_status = {
        "DATABASE_URL": mask_secret(os.environ.get("DATABASE_URL", "")),
        "API_AUTH_KEY": mask_secret(os.environ.get("API_AUTH_KEY", "")),
        "GEMINI_API_KEY": mask_secret(os.environ.get("GEMINI_API_KEY", "")),
        "OPENAI_API_KEY": mask_secret(os.environ.get("OPENAI_API_KEY", "")),
        "ASSEMBLYAI_API_KEY": mask_secret(os.environ.get("ASSEMBLYAI_API_KEY", "")),
    }
    
    return {
        "status": "healthy",
        "service": "audio-rag-pipeline", 
        "timestamp": datetime.utcnow().isoformat(),
        "secrets": secret_status,
        "assemblyai_ready": os.environ.get("ASSEMBLYAI_API_KEY") is not None
    }


# Setup endpoint
@app.function(image=image, cpu=1.0, secrets=SECRETS)
@modal.fastapi_endpoint(method="POST", label="setup")
def setup_endpoint(authorization: str = Header(None)) -> Dict[str, Any]:
    """Initialize database (run this once)"""
    try:
        validate_api_key(authorization)
        
        result = setup_database.remote()
        return result
    except Exception as e:
        return {"error": str(e), "status": "error"}


# Business Intelligence endpoint (consolidated)
@app.function(image=image, cpu=1.0, secrets=SECRETS)
@modal.fastapi_endpoint(method="GET", label="business-intelligence")
def business_intelligence_endpoint(
    action: str,
    stage: str = None,
    client: str = None,
    min_probability: int = 70,
    limit: int = 100,
    offset: int = 0,
    authorization: str = Header(None)
) -> Dict[str, Any]:
    """Consolidated business intelligence endpoint"""
    logger.info("📈 BUSINESS_INTELLIGENCE: Received request - action=%s, limit=%d, offset=%d", action, limit, offset)
    try:
        validate_api_key(authorization)
        logger.info("📈 BUSINESS_INTELLIGENCE: Authorization validated")
        
        if action == "deals":
            results = get_deals_by_stage.remote(stage)
            return {"status": "success", "deals": results}
        elif action == "high-probability-deals":
            results = get_high_probability_deals.remote(min_probability)
            return {"status": "success", "deals": results}
        elif action == "client-insights":
            if not client:
                return {"status": "error", "error": "client parameter required"}
            results = get_client_insights.remote(client)
            return {"status": "success", **results}
        elif action == "competitors":
            results = get_competitor_analysis.remote()
            return {"status": "success", **results}
        elif action == "action-items":
            results = get_action_items_summary.remote()
            return {"status": "success", **results}
        elif action == "all-meetings":
            logger.info("📈 BUSINESS_INTELLIGENCE: Calling get_all_meetings_with_metadata with limit=%d, offset=%d", limit, offset)
            results = get_all_meetings_with_metadata.remote(limit, offset)
            logger.info("📈 BUSINESS_INTELLIGENCE: Received results from get_all_meetings_with_metadata")
            return {"status": "success", **results}
        else:
            return {"status": "error", "error": "Invalid action parameter"}
            
    except Exception as e:
        return {"status": "error", "error": str(e)}


# Business Intelligence Query Functions
@app.function(
    image=image,
    cpu=1.0,
    secrets=SECRETS
)
async def get_deals_by_stage(stage: str = None) -> List[Dict[str, Any]]:
    """Get all deals filtered by stage"""
    import asyncpg
    import os
    
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    
    try:
        if stage:
            query = """
                SELECT id, filename, created_at,
                       meeting_metadata->>'participants' as participants,
                       meeting_metadata->'sales_intelligence'->>'deal_stage' as deal_stage,
                       meeting_metadata->'sales_intelligence'->>'probability_to_close' as probability,
                       meeting_metadata->'sales_intelligence'->>'estimated_deal_value' as deal_value,
                       meeting_metadata->'business_intelligence'->>'timeline_urgency' as urgency
                FROM audio_transcripts
                WHERE meeting_metadata->'sales_intelligence'->>'deal_stage' = $1
                ORDER BY created_at DESC
            """
            results = await conn.fetch(query, stage)
        else:
            query = """
                SELECT id, filename, created_at,
                       meeting_metadata->>'participants' as participants,
                       meeting_metadata->'sales_intelligence'->>'deal_stage' as deal_stage,
                       meeting_metadata->'sales_intelligence'->>'probability_to_close' as probability,
                       meeting_metadata->'sales_intelligence'->>'estimated_deal_value' as deal_value,
                       meeting_metadata->'business_intelligence'->>'timeline_urgency' as urgency
                FROM audio_transcripts
                WHERE meeting_metadata IS NOT NULL
                ORDER BY created_at DESC
            """
            results = await conn.fetch(query)
            
        return [dict(record) for record in results]
    
    finally:
        await conn.close()


@app.function(
    image=image,
    cpu=1.0,
    secrets=SECRETS
)
async def get_high_probability_deals(min_probability: int = 70) -> List[Dict[str, Any]]:
    """Get deals with high close probability"""
    import asyncpg
    import os
    
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    
    try:
        query = """
            SELECT id, filename, created_at, summary,
                   meeting_metadata->'participants'->>'client_company_name' as client_company,
                   meeting_metadata->'sales_intelligence'->>'probability_to_close' as probability,
                   meeting_metadata->'sales_intelligence'->>'estimated_deal_value' as deal_value,
                   meeting_metadata->'action_items'->>'next_steps' as next_steps,
                   meeting_metadata->'insights'->>'green_flags' as green_flags
            FROM audio_transcripts
            WHERE (meeting_metadata->'sales_intelligence'->>'probability_to_close')::int >= $1
            ORDER BY (meeting_metadata->'sales_intelligence'->>'probability_to_close')::int DESC
        """
        results = await conn.fetch(query, min_probability)
        
        return [dict(record) for record in results]
    
    finally:
        await conn.close()


@app.function(
    image=image,
    cpu=1.0,
    secrets=SECRETS
)
async def get_client_insights(client_company: str) -> Dict[str, Any]:
    """Get comprehensive insights for a specific client"""
    import asyncpg
    import os
    import json
    
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    
    try:
        query = """
            SELECT id, filename, created_at, summary, transcription,
                   meeting_metadata
            FROM audio_transcripts
            WHERE meeting_metadata->'participants'->>'client_company_name' ILIKE $1
            ORDER BY created_at DESC
        """
        results = await conn.fetch(query, f"%{client_company}%")
        
        if not results:
            return {"error": "No meetings found for this client"}
        
        # Aggregate insights across all meetings
        all_problems = []
        all_needs = []
        all_objections = []
        all_action_items = []
        meeting_history = []
        
        for record in results:
            # Parse metadata - handle both string (JSON) and dict (JSONB) cases
            metadata = record['meeting_metadata']
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata) if metadata else {}
                except json.JSONDecodeError:
                    metadata = {}
            elif not isinstance(metadata, dict):
                metadata = {}
            
            meeting_history.append({
                "date": record['created_at'].isoformat(),
                "type": metadata.get('meeting_context', {}).get('meeting_type'),
                "sentiment": metadata.get('meeting_context', {}).get('meeting_sentiment'),
                "stage": metadata.get('sales_intelligence', {}).get('deal_stage')
            })
            
            if 'business_intelligence' in metadata:
                all_problems.extend(metadata['business_intelligence'].get('client_problems', []))
                all_needs.extend(metadata['business_intelligence'].get('client_needs', []))
                
            if 'sales_intelligence' in metadata:
                all_objections.extend(metadata['sales_intelligence'].get('objections_raised', []))
                
            if 'action_items' in metadata:
                all_action_items.extend(metadata['action_items'].get('consultant_action_items', []))
        
        return {
            "client_company": client_company,
            "total_meetings": len(results),
            "meeting_history": meeting_history,
            "all_problems": all_problems,
            "all_needs": all_needs,
            "all_objections": all_objections,
            "pending_action_items": all_action_items,
            "latest_metadata": results[0]['meeting_metadata'] if results else {}
        }
    
    finally:
        await conn.close()


@app.function(
    image=image,
    cpu=1.0,
    secrets=SECRETS
)
async def get_competitor_analysis() -> Dict[str, Any]:
    """Analyze competitor mentions across all meetings"""
    import asyncpg
    import os
    from collections import Counter
    
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    
    try:
        query = """
            SELECT meeting_metadata->'competitive_landscape'->>'competitors_mentioned' as competitors,
                   meeting_metadata->'competitive_landscape'->>'competitive_advantages' as advantages,
                   meeting_metadata->'competitive_landscape'->>'competitive_concerns' as concerns
            FROM audio_transcripts
            WHERE meeting_metadata->'competitive_landscape' IS NOT NULL
        """
        results = await conn.fetch(query)
        
        competitor_mentions = Counter()
        all_advantages = []
        all_concerns = []
        
        for record in results:
            if record['competitors']:
                import json
                competitors = json.loads(record['competitors'])
                for comp in competitors:
                    competitor_mentions[comp] += 1
                    
            if record['advantages']:
                advantages = json.loads(record['advantages'])
                all_advantages.extend(advantages)
                
            if record['concerns']:
                concerns = json.loads(record['concerns'])
                all_concerns.extend(concerns)
        
        return {
            "competitor_frequency": dict(competitor_mentions),
            "our_advantages": list(set(all_advantages)),
            "competitive_concerns": list(set(all_concerns)),
            "total_meetings_with_competitors": len(results)
        }
    
    finally:
        await conn.close()


@app.function(
    image=image,
    cpu=1.0,
    secrets=SECRETS
)
async def get_action_items_summary() -> Dict[str, Any]:
    """Get all pending action items across meetings"""
    import asyncpg
    import os
    import json
    from datetime import datetime
    
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    
    try:
        query = """
            SELECT filename, created_at,
                   meeting_metadata->'participants'->>'client_company_name' as client,
                   meeting_metadata->'action_items' as action_items
            FROM audio_transcripts
            WHERE meeting_metadata->'action_items' IS NOT NULL
            ORDER BY created_at DESC
        """
        results = await conn.fetch(query)
        
        consultant_tasks = []
        client_tasks = []
        milestones = []
        
        for record in results:
            action_items = record['action_items']
            
            # Add client context to each task
            if 'consultant_action_items' in action_items:
                for task in action_items['consultant_action_items']:
                    task['client'] = record['client']
                    task['meeting_date'] = record['created_at'].isoformat()
                    consultant_tasks.append(task)
                    
            if 'client_action_items' in action_items:
                for task in action_items['client_action_items']:
                    task['client'] = record['client']
                    task['meeting_date'] = record['created_at'].isoformat()
                    client_tasks.append(task)
                    
            if 'key_milestones' in action_items:
                for milestone in action_items['key_milestones']:
                    milestone['client'] = record['client']
                    milestones.append(milestone)
        
        return {
            "consultant_action_items": consultant_tasks,
            "client_action_items": client_tasks,
            "upcoming_milestones": milestones,
            "summary": {
                "total_consultant_tasks": len(consultant_tasks),
                "total_client_tasks": len(client_tasks),
                "total_milestones": len(milestones)
            }
        }
    
    finally:
        await conn.close()


# Raw function that can be called directly for debugging
async def _raw_get_all_meetings_with_metadata(limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    """Get all meetings with complete metadata for review - RAW VERSION FOR DEBUGGING"""
    import asyncpg
    import os
    import json
    
    logger.info("📊 GET_ALL_MEETINGS: Starting query with limit=%d, offset=%d", limit, offset)
    
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    
    try:
        # Get total count
        count_query = "SELECT COUNT(*) FROM audio_transcripts"
        total_count = await conn.fetchval(count_query)
        logger.info("📊 GET_ALL_MEETINGS: Total records in database: %d", total_count)
        
        # Get paginated results
        query = """
            SELECT id, filename, transcription, summary, segments, 
                   meeting_metadata, language, created_at, updated_at
            FROM audio_transcripts
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
        """
        results = await conn.fetch(query, limit, offset)
        logger.info("📊 GET_ALL_MEETINGS: Retrieved %d records", len(results))
        
        meetings = []
        for i, record in enumerate(results):
            logger.info("📊 GET_ALL_MEETINGS: Processing record %d - filename: %s", i+1, record.get('filename', 'unknown'))
            meeting = {
                "id": str(record['id']),
                "filename": record['filename'],
                "transcription": record['transcription'],
                "summary": record['summary'],
                "segments": record['segments'] if record['segments'] else [],
                "meeting_metadata": record['meeting_metadata'] if record['meeting_metadata'] else {},
                "language": record['language'],
                "created_at": record['created_at'].isoformat(),
                "updated_at": record['updated_at'].isoformat() if record['updated_at'] else None
            }
            
            # Extract key fields for quick overview
            if record['meeting_metadata']:
                # Parse metadata - handle both string (JSON) and dict (JSONB) cases
                metadata = record['meeting_metadata']
                logger.info("📊 GET_ALL_MEETINGS: Raw metadata type: %s, content: %s", type(metadata), str(metadata)[:200])
                
                # Handle case where JSONB is returned as string
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata) if metadata else {}
                        logger.info("📊 GET_ALL_MEETINGS: Parsed string to dict successfully")
                    except json.JSONDecodeError as e:
                        logger.error("📊 GET_ALL_MEETINGS: Failed to parse JSON metadata: %s", e)
                        metadata = {}
                elif not isinstance(metadata, dict):
                    logger.warning("📊 GET_ALL_MEETINGS: Unexpected metadata type: %s, converting to empty dict", type(metadata))
                    metadata = {}
                
                logger.info("📊 GET_ALL_MEETINGS: Final metadata type: %s, keys: %s", type(metadata), list(metadata.keys()) if metadata else [])
                
                meeting['quick_view'] = {
                    "client": metadata.get('participants', {}).get('client_company_name') if metadata.get('participants') else None,
                    "meeting_type": metadata.get('meeting_context', {}).get('meeting_type') if metadata.get('meeting_context') else None,
                    "sentiment": metadata.get('meeting_context', {}).get('meeting_sentiment') if metadata.get('meeting_context') else None,
                    "deal_stage": metadata.get('sales_intelligence', {}).get('deal_stage') if metadata.get('sales_intelligence') else None,
                    "probability": metadata.get('sales_intelligence', {}).get('probability_to_close') if metadata.get('sales_intelligence') else None,
                    "next_steps": metadata.get('action_items', {}).get('next_steps', []) if metadata.get('action_items') else []
                }
                
                # Update the meeting_metadata field with parsed version
                meeting['meeting_metadata'] = metadata
            
            meetings.append(meeting)
        
        result = {
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "meetings": meetings,
            "has_more": (offset + limit) < total_count
        }
        logger.info("📊 GET_ALL_MEETINGS: Returning %d meetings, total_count: %d", len(meetings), total_count)
        return result
    
    finally:
        await conn.close()


# Modal-decorated wrapper that calls the raw function
@app.function(
    image=image,
    cpu=1.0,
    secrets=SECRETS
)
async def get_all_meetings_with_metadata(limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    """Get all meetings with complete metadata for review - MODAL WRAPPER"""
    return await _raw_get_all_meetings_with_metadata(limit, offset)


if __name__ == "__main__":
    print("Deploy with: modal deploy main.py")
    print("Don't forget to run the /setup endpoint once after deployment!")
    print("\nEnhancements in this version:")
    print("- AssemblyAI for transcription and speaker diarization")
    print("- Structured metadata extraction")
    print("- Clean audio_transcripts table with enhanced schema")