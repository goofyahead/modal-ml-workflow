import streamlit as st
import requests
import json
import pandas as pd
import os
from datetime import datetime

# Configuration
MODAL_WORKSPACE = "goofyahead"
MODAL_APP = "audio-rag-pipeline" 
API_KEY = os.getenv("API_AUTH_KEY", "")  # Get from environment

# Set page config
st.set_page_config(
    page_title="Audio RAG Business Intelligence",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

def make_authenticated_request(endpoint, method="GET", **kwargs):
    """Make authenticated request to Modal endpoints"""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    # Construct proper Modal endpoint URL
    url = f"https://{MODAL_WORKSPACE}--{endpoint}.modal.run"
    
    if method == "GET":
        response = requests.get(url, headers=headers, **kwargs)
    elif method == "POST":
        response = requests.post(url, headers=headers, **kwargs)
    
    return response

def upload_audio_file(audio_file):
    """Upload audio file to Modal for processing"""
    files = {"audio_file": audio_file}
    data = {"authorization": f"Bearer {API_KEY}"}
    
    with st.spinner("🔄 Processing audio with AssemblyAI & AI Intelligence... This may take a few minutes."):
        url = f"https://{MODAL_WORKSPACE}--upload-audio.modal.run"
        response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

def search_transcripts(query, search_type="transcription", limit=5):
    """Search through transcripts"""
    params = {"q": query, "type": search_type, "limit": limit}
    
    response = make_authenticated_request("search", "GET", params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Search error: {response.status_code} - {response.text}")
        return None

def ask_question(question, context_limit=3):
    """Ask question using RAG"""
    params = {"q": question, "context_limit": context_limit}
    
    response = make_authenticated_request("ask", "GET", params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"RAG error: {response.status_code} - {response.text}")
        return None

def get_deals(stage=None):
    """Get deals by stage"""
    params = {"action": "deals"}
    if stage:
        params["stage"] = stage
    response = make_authenticated_request("business-intelligence", "GET", params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching deals: {response.status_code}")
        return None

def get_high_probability_deals(min_prob=70):
    """Get high probability deals"""
    params = {"action": "high-probability-deals", "min_probability": min_prob}
    response = make_authenticated_request("business-intelligence", "GET", params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching high probability deals: {response.status_code}")
        return None

def get_client_insights(client_name):
    """Get insights for a specific client"""
    params = {"action": "client-insights", "client": client_name}
    response = make_authenticated_request("business-intelligence", "GET", params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching client insights: {response.status_code}")
        return None

def get_competitor_analysis():
    """Get competitor analysis"""
    params = {"action": "competitors"}
    response = make_authenticated_request("business-intelligence", "GET", params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching competitor analysis: {response.status_code}")
        return None

def get_action_items():
    """Get all action items"""
    params = {"action": "action-items"}
    response = make_authenticated_request("business-intelligence", "GET", params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching action items: {response.status_code}")
        return None

def get_all_meetings(limit=100, offset=0):
    """Get all meetings with metadata"""
    params = {"action": "all-meetings", "limit": limit, "offset": offset}
    response = make_authenticated_request("business-intelligence", "GET", params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching meetings: {response.status_code}")
        return None

def test_connection():
    """Test connection to Modal health endpoint"""
    response = make_authenticated_request("health", "GET")
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Health check failed: {response.status_code} - {response.text}")
        return None

def display_metadata(metadata):
    """Display meeting metadata in a nice format"""
    if not metadata:
        return
    
    # Meeting Context
    if "meeting_context" in metadata:
        ctx = metadata["meeting_context"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Meeting Type", ctx.get("meeting_type", "N/A"))
        with col2:
            sentiment = ctx.get("meeting_sentiment", "N/A")
            color = "🟢" if "positive" in sentiment else "🔴" if "concerned" in sentiment else "🟡"
            st.metric("Sentiment", f"{color} {sentiment}")
        with col3:
            st.metric("Stage", ctx.get("meeting_stage", "N/A"))
    
    # Sales Intelligence
    if "sales_intelligence" in metadata:
        sales = metadata["sales_intelligence"]
        st.subheader("📊 Sales Intelligence")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Deal Stage", sales.get("deal_stage", "N/A"))
        with col2:
            prob = sales.get("probability_to_close", "0")
            st.metric("Close Probability", f"{prob}%")
        with col3:
            st.metric("Deal Value", sales.get("estimated_deal_value", "N/A"))
        
        if sales.get("buying_signals"):
            st.success("✅ Buying Signals: " + ", ".join(sales["buying_signals"]))
        if sales.get("objections_raised"):
            st.warning("⚠️ Objections: " + ", ".join(sales["objections_raised"]))
    
    # Client Info
    if "participants" in metadata:
        st.subheader("🏢 Client Information")
        st.write(f"**Company:** {metadata['participants'].get('client_company_name', 'N/A')}")
        if metadata['participants'].get('client_team'):
            team = metadata['participants']['client_team']
            st.write("**Client Team:**")
            for person in team:
                st.write(f"- {person.get('name', 'Unknown')} ({person.get('role', 'N/A')})")
    
    # Action Items
    if "action_items" in metadata:
        actions = metadata["action_items"]
        if actions.get("next_steps"):
            st.subheader("📋 Next Steps")
            for step in actions["next_steps"]:
                st.write(f"- {step}")

# Streamlit UI
st.title("🎧 Audio RAG Business Intelligence Platform")
st.markdown("**Consultant Meeting Intelligence System** - Transform conversations into insights")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
workspace = st.sidebar.text_input("Modal Workspace", value=MODAL_WORKSPACE)
app_name = st.sidebar.text_input("Modal App Name", value=MODAL_APP)
api_key_input = st.sidebar.text_input("API Key", value=API_KEY, type="password", help="Your Modal API authentication key")

# Update globals if changed
if workspace != MODAL_WORKSPACE:
    MODAL_WORKSPACE = workspace
if app_name != MODAL_APP:
    MODAL_APP = app_name
if api_key_input != API_KEY:
    API_KEY = api_key_input

# Health check in sidebar
if st.sidebar.button("🩺 Test Connection"):
    if not API_KEY:
        st.sidebar.error("Please set your API key first")
    else:
        with st.spinner("Testing connection..."):
            result = test_connection()
            if result and result.get("status") == "healthy":
                st.sidebar.success(f"✅ Connected! Service: {result.get('service')}")
            else:
                st.sidebar.error("❌ Connection failed")

# Initialize database button
if st.sidebar.button("🗄️ Initialize Database"):
    if not API_KEY:
        st.sidebar.error("Please set your API key first")
    else:
        with st.spinner("Initializing database..."):
            response = make_authenticated_request("setup", "POST")
            if response.status_code == 200:
                st.sidebar.success("✅ Database initialized!")
            else:
                st.sidebar.error("❌ Database initialization failed")

st.sidebar.markdown("---")
st.sidebar.caption("Enhanced with WhisperX speaker diarization and structured meeting intelligence")

# Main interface with tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📁 Upload", 
    "🔍 Search", 
    "❓ Ask", 
    "💼 Deals Pipeline",
    "🏢 Client Insights",
    "🎯 Competitors",
    "✅ Action Items",
    "📊 All Meetings"
])

with tab1:
    st.header("Upload Audio File")
    st.markdown("Process consultant-client meetings with AI-powered intelligence extraction")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg', 'webm'],
        help="Supports WAV, MP3, MP4, M4A, FLAC, OGG, WEBM formats (max 50MB)"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**File:** {uploaded_file.name}")
        with col2:
            st.write(f"**Size:** {uploaded_file.size / 1024 / 1024:.2f} MB")
        
        if st.button("🚀 Process with AI Intelligence", type="primary"):
            if not API_KEY:
                st.error("Please set your API key in the sidebar")
            else:
                result = upload_audio_file(uploaded_file)
                
                if result and (result.get("status") == "success" or result.get("status") == "completed"):
                    st.success("✅ Audio processed successfully with speaker diarization!")
                    
                    # Display results in tabs
                    res_tab1, res_tab2, res_tab3 = st.tabs(["Transcription", "Summary", "Intelligence"])
                    
                    with res_tab1:
                        st.subheader("🎤 Speaker-Aware Transcription")
                        st.text_area("", value=result.get("transcription", ""), height=300, disabled=True)
                    
                    with res_tab2:
                        st.subheader("📄 Summary")
                        st.write(result.get("summary", ""))
                    
                    with res_tab3:
                        st.subheader("📊 Meeting Intelligence")
                        if result.get("metadata"):
                            display_metadata(result["metadata"])
                        else:
                            st.info("Metadata extraction in progress...")
                    
                    st.info(f"Record ID: {result.get('record_id')}")

with tab2:
    st.header("Search Transcripts")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_query = st.text_input("Search query", placeholder="Enter keywords to search for...")
    with col2:
        search_type = st.selectbox("Search in", ["transcription", "summary"])
    with col3:
        search_limit = st.number_input("Max results", 1, 20, 5)
    
    if st.button("🔍 Search", type="primary") and search_query:
        if not API_KEY:
            st.error("Please set your API key in the sidebar")
        else:
            results = search_transcripts(search_query, search_type, search_limit)
            
            if results and results.get("status") == "success":
                search_results = results.get("results", [])
                
                if search_results:
                    st.subheader(f"Found {len(search_results)} results")
                    
                    for result in search_results:
                        with st.expander(f"📁 {result['filename']} (Similarity: {result.get('transcription_similarity', 0):.1%})"):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write("**Transcription:**")
                                st.write(result['transcription'][:500] + "..." if len(result['transcription']) > 500 else result['transcription'])
                            with col2:
                                if result.get('meeting_metadata'):
                                    metadata = result['meeting_metadata']
                                    st.write("**Quick Info:**")
                                    st.write(f"Client: {metadata.get('participants', {}).get('client_company_name', 'N/A')}")
                                    st.write(f"Stage: {metadata.get('sales_intelligence', {}).get('deal_stage', 'N/A')}")
                            
                            if result.get('summary'):
                                st.write("**Summary:**")
                                st.write(result['summary'])
                            
                            st.caption(f"Created: {result['created_at']}")
                else:
                    st.info("No results found")

with tab3:
    st.header("Ask Questions")
    
    question = st.text_area(
        "What would you like to know about your meetings?",
        placeholder="e.g., What are the main objections from clients? What features are most requested?",
        height=100
    )
    
    context_limit = st.slider("Context sources", 1, 10, 3, 
                             help="Number of relevant transcripts to use as context")
    
    if st.button("💡 Get Answer", type="primary") and question:
        if not API_KEY:
            st.error("Please set your API key in the sidebar")
        else:
            with st.spinner("Analyzing your meetings..."):
                result = ask_question(question, context_limit)
            
            if result and result.get("status") == "success":
                st.subheader("Answer")
                st.write(result.get("answer", ""))
                
                # Show sources
                sources = result.get("sources", [])
                if sources:
                    st.subheader("📚 Sources")
                    for source in sources:
                        st.caption(f"📁 {source['filename']} (Relevance: {source['similarity']:.1%})")

with tab4:
    st.header("💼 Sales Pipeline Management")
    
    # Deal stage buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("All Deals"):
            st.session_state.deal_stage = None
    with col2:
        if st.button("Prospecting"):
            st.session_state.deal_stage = "prospecting"
    with col3:
        if st.button("Qualification"):
            st.session_state.deal_stage = "qualification"
    with col4:
        if st.button("Proposal"):
            st.session_state.deal_stage = "proposal"
    with col5:
        if st.button("🔥 Hot Deals"):
            st.session_state.deal_stage = "hot"
    
    # Get deals
    if st.button("Load Deals", type="primary"):
        if not API_KEY:
            st.error("Please set your API key in the sidebar")
        else:
            stage = st.session_state.get('deal_stage')
            
            if stage == "hot":
                result = get_high_probability_deals()
                if result and result.get("status") == "success":
                    deals = result.get("deals", [])
                    
                    if deals:
                        st.subheader(f"🔥 High Probability Deals ({len(deals)})")
                        
                        for deal in deals:
                            with st.expander(f"{deal.get('client_company', 'Unknown')} - {deal.get('probability', 0)}%"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Probability", f"{deal.get('probability', 0)}%")
                                    st.metric("Deal Value", deal.get('deal_value', 'N/A'))
                                with col2:
                                    st.write("**Summary:**")
                                    st.write(deal.get('summary', 'N/A'))
                                
                                if deal.get('green_flags'):
                                    st.success("✅ " + deal['green_flags'])
                                if deal.get('next_steps'):
                                    st.info("Next: " + deal['next_steps'])
            else:
                result = get_deals(stage)
                if result and result.get("status") == "success":
                    deals = result.get("deals", [])
                    
                    if deals:
                        # Calculate metrics
                        total_deals = len(deals)
                        high_prob = len([d for d in deals if int(d.get('probability', 0)) >= 70])
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Deals", total_deals)
                        with col2:
                            st.metric("High Probability", high_prob)
                        with col3:
                            avg_prob = sum(int(d.get('probability', 0)) for d in deals) / len(deals) if deals else 0
                            st.metric("Avg Probability", f"{avg_prob:.0f}%")
                        
                        # Display deals
                        st.subheader("Deal Pipeline")
                        
                        # Create DataFrame for better display
                        df_data = []
                        for deal in deals:
                            df_data.append({
                                "File": deal.get('filename', 'N/A')[:30] + "...",
                                "Stage": deal.get('deal_stage', 'N/A'),
                                "Probability": f"{deal.get('probability', 0)}%",
                                "Value": deal.get('deal_value', 'N/A'),
                                "Urgency": deal.get('urgency', 'N/A')
                            })
                        
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True)

with tab5:
    st.header("🏢 Client Insights")
    
    client_name = st.text_input("Client Company Name", placeholder="Enter client company name...")
    
    if st.button("Get Insights", type="primary") and client_name:
        if not API_KEY:
            st.error("Please set your API key in the sidebar")
        else:
            with st.spinner("Analyzing client history..."):
                result = get_client_insights(client_name)
            
            if result and result.get("status") == "success":
                st.subheader(f"Client Profile: {result.get('client_company', client_name)}")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Meetings", result.get('total_meetings', 0))
                with col2:
                    if result.get('meeting_history'):
                        last_meeting = result['meeting_history'][0]
                        st.metric("Last Meeting", last_meeting.get('type', 'N/A'))
                with col3:
                    if result.get('meeting_history'):
                        sentiments = [m.get('sentiment', '') for m in result['meeting_history']]
                        positive = sum(1 for s in sentiments if 'positive' in s)
                        st.metric("Positive Meetings", f"{positive}/{len(sentiments)}")
                
                # Meeting Timeline
                if result.get('meeting_history'):
                    st.subheader("📅 Meeting Timeline")
                    for meeting in result['meeting_history']:
                        date = datetime.fromisoformat(meeting['date']).strftime('%Y-%m-%d')
                        sentiment_emoji = "🟢" if "positive" in meeting.get('sentiment', '') else "🔴" if "concerned" in meeting.get('sentiment', '') else "🟡"
                        st.write(f"{date} - {meeting.get('type', 'N/A')} {sentiment_emoji} - Stage: {meeting.get('stage', 'N/A')}")
                
                # Problems & Needs
                col1, col2 = st.columns(2)
                with col1:
                    if result.get('all_problems'):
                        st.subheader("🔴 Problems Identified")
                        for problem in result['all_problems']:
                            if isinstance(problem, dict):
                                st.write(f"- {problem.get('problem', problem)}")
                            else:
                                st.write(f"- {problem}")
                
                with col2:
                    if result.get('all_needs'):
                        st.subheader("🎯 Client Needs")
                        for need in result['all_needs']:
                            st.write(f"- {need}")
                
                # Objections
                if result.get('all_objections'):
                    st.subheader("⚠️ Objections Raised")
                    for objection in result['all_objections']:
                        st.warning(objection)
                
                # Action Items
                if result.get('pending_action_items'):
                    st.subheader("📋 Pending Action Items")
                    for item in result['pending_action_items']:
                        if isinstance(item, dict):
                            st.write(f"- {item.get('task', item)}")
                        else:
                            st.write(f"- {item}")

with tab6:
    st.header("🎯 Competitive Intelligence")
    
    if st.button("Analyze Competitors", type="primary"):
        if not API_KEY:
            st.error("Please set your API key in the sidebar")
        else:
            with st.spinner("Analyzing competitive landscape..."):
                result = get_competitor_analysis()
            
            if result and result.get("status") == "success":
                st.subheader("Competitive Landscape Analysis")
                
                # Competitor frequency
                if result.get('competitor_frequency'):
                    st.subheader("📊 Competitor Mentions")
                    comp_df = pd.DataFrame(
                        list(result['competitor_frequency'].items()),
                        columns=['Competitor', 'Mentions']
                    ).sort_values('Mentions', ascending=False)
                    
                    st.bar_chart(comp_df.set_index('Competitor'))
                
                # Our advantages
                col1, col2 = st.columns(2)
                with col1:
                    if result.get('our_advantages'):
                        st.subheader("✅ Our Competitive Advantages")
                        for adv in result['our_advantages']:
                            st.success(adv)
                
                with col2:
                    if result.get('competitive_concerns'):
                        st.subheader("⚠️ Competitive Concerns")
                        for concern in result['competitive_concerns']:
                            st.warning(concern)
                
                st.metric("Meetings with competitor mentions", 
                         result.get('total_meetings_with_competitors', 0))

with tab7:
    st.header("✅ Action Items Management")
    
    if st.button("Load Action Items", type="primary"):
        if not API_KEY:
            st.error("Please set your API key in the sidebar")
        else:
            with st.spinner("Loading action items..."):
                result = get_action_items()
            
            if result and result.get("status") == "success":
                # Display metrics
                if result.get('summary'):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Our Tasks", result['summary']['total_consultant_tasks'])
                    with col2:
                        st.metric("Client Tasks", result['summary']['total_client_tasks'])
                    with col3:
                        st.metric("Milestones", result['summary']['total_milestones'])
                
                # Consultant action items
                if result.get('consultant_action_items'):
                    st.subheader("📋 Our Action Items")
                    for item in result['consultant_action_items']:
                        with st.expander(f"{item.get('client', 'Unknown')} - {item.get('task', 'N/A')[:50]}..."):
                            st.write(f"**Task:** {item.get('task', 'N/A')}")
                            st.write(f"**Client:** {item.get('client', 'N/A')}")
                            if item.get('owner'):
                                st.write(f"**Owner:** {item['owner']}")
                            if item.get('deadline'):
                                st.write(f"**Deadline:** {item['deadline']}")
                            st.write(f"**Meeting Date:** {item.get('meeting_date', 'N/A')}")
                
                # Client action items
                if result.get('client_action_items'):
                    st.subheader("📋 Client Action Items")
                    for item in result['client_action_items']:
                        with st.expander(f"{item.get('client', 'Unknown')} - {item.get('task', 'N/A')[:50]}..."):
                            st.write(f"**Task:** {item.get('task', 'N/A')}")
                            st.write(f"**Client:** {item.get('client', 'N/A')}")
                            if item.get('owner'):
                                st.write(f"**Owner:** {item['owner']}")
                            if item.get('deadline'):
                                st.write(f"**Deadline:** {item['deadline']}")
                
                # Milestones
                if result.get('upcoming_milestones'):
                    st.subheader("🎯 Upcoming Milestones")
                    milestone_data = []
                    for milestone in result['upcoming_milestones']:
                        milestone_data.append({
                            "Client": milestone.get('client', 'N/A'),
                            "Date": milestone.get('date', 'N/A'),
                            "Deliverable": milestone.get('deliverable', 'N/A'),
                            "Responsible": milestone.get('responsible_party', 'N/A')
                        })
                    
                    if milestone_data:
                        df = pd.DataFrame(milestone_data)
                        st.dataframe(df, use_container_width=True)

with tab8:
    st.header("📊 All Meetings Database")
    
    col1, col2 = st.columns(2)
    with col1:
        limit = st.selectbox("Results per page", [10, 25, 50, 100])
    with col2:
        offset = st.number_input("Skip records", min_value=0, value=0, step=limit)
    
    if st.button("Load Meetings", type="primary"):
        if not API_KEY:
            st.error("Please set your API key in the sidebar")
        else:
            with st.spinner("Loading meetings..."):
                result = get_all_meetings(limit, offset)
            
            if result and result.get("status") == "success":
                st.subheader(f"Meetings ({offset + 1}-{min(offset + limit, result['total_count'])} of {result['total_count']})")
                
                meetings = result.get('meetings', [])
                
                if meetings:
                    for meeting in meetings:
                        quick = meeting.get('quick_view', {})
                        
                        with st.expander(f"📁 {meeting['filename']} - {quick.get('client', 'Unknown')}"):
                            # Quick info
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Type:** {quick.get('meeting_type', 'N/A')}")
                                st.write(f"**Language:** {meeting.get('language', 'N/A')}")
                            with col2:
                                st.write(f"**Sentiment:** {quick.get('sentiment', 'N/A')}")
                                st.write(f"**Stage:** {quick.get('deal_stage', 'N/A')}")
                            with col3:
                                st.write(f"**Probability:** {quick.get('probability', 'N/A')}")
                                st.write(f"**Created:** {meeting.get('created_at', 'N/A')}")
                            
                            # Summary
                            if meeting.get('summary'):
                                st.subheader("Summary")
                                st.write(meeting['summary'])
                            
                            # Metadata
                            if meeting.get('meeting_metadata'):
                                with st.expander("View Full Metadata"):
                                    st.json(meeting['meeting_metadata'])
                            
                            # Segments
                            if meeting.get('segments') and len(meeting['segments']) > 0:
                                with st.expander(f"Speaker Segments ({len(meeting['segments'])})"):
                                    for i, seg in enumerate(meeting['segments'][:5]):
                                        st.write(f"**{seg.get('speaker', 'Unknown')}:** {seg.get('text', '')}")
                                    if len(meeting['segments']) > 5:
                                        st.write("...")
                            
                            # Full transcription
                            with st.expander("View Full Transcription"):
                                st.text_area("", value=meeting.get('transcription', ''), height=300, disabled=True)
                
                # Pagination
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if offset > 0:
                        if st.button("← Previous"):
                            st.rerun()
                with col3:
                    if result.get('has_more'):
                        if st.button("Next →"):
                            st.rerun()

# Footer
st.markdown("---")
st.caption("🎧 Powered by Modal, WhisperX, OpenAI, and Gemini | Enhanced with Business Intelligence")
st.caption("Transform your consultant-client meetings into actionable insights")