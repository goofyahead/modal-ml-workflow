# Audio RAG Pipeline - Development Commands
# Usage: make <command>

# Configuration
API_KEY = c74555c8-15e5-4bff-a331-baf9f4b4113f
STREAMLIT_PORT = 8501
MODAL_APP = main.py

# Default target
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make run          - Run Streamlit UI (enhanced version)"
	@echo "  make run-basic    - Run basic Streamlit UI"
	@echo "  make deploy       - Deploy to Modal"
	@echo "  make health       - Check Modal health"
	@echo "  make logs         - View Modal logs"
	@echo "  make test-upload  - Test audio upload endpoint"
	@echo "  make setup-db     - Initialize database"
	@echo "  make clean        - Kill running Streamlit processes"
	@echo "  make install      - Install Python dependencies"
	@echo ""
	@echo "UI Access:"
	@echo "  Local:   http://localhost:$(STREAMLIT_PORT)"
	@echo "  Network: http://192.168.0.11:$(STREAMLIT_PORT)"

# Run enhanced Streamlit UI
.PHONY: run
run:
	@echo "🚀 Starting Enhanced Streamlit UI..."
	@echo "📡 API Key: $(API_KEY)"
	@echo "🌐 Access: http://localhost:$(STREAMLIT_PORT)"
	API_AUTH_KEY=$(API_KEY) streamlit run streamlit_ui_enhanced.py --server.port $(STREAMLIT_PORT)

# Run basic Streamlit UI  
.PHONY: run-basic
run-basic:
	@echo "🚀 Starting Basic Streamlit UI..."
	API_AUTH_KEY=$(API_KEY) streamlit run streamlit_ui.py --server.port $(STREAMLIT_PORT)

# Deploy to Modal
.PHONY: deploy
deploy:
	@echo "🚀 Deploying to Modal..."
	modal deploy $(MODAL_APP)

# Check Modal health
.PHONY: health
health:
	@echo "🩺 Checking Modal health..."
	curl -H "Authorization: Bearer $(API_KEY)" https://goofyahead--health.modal.run

# View Modal logs
.PHONY: logs
logs:
	@echo "📋 Viewing Modal logs..."
	modal app logs audio-rag-pipeline

# Test upload endpoint
.PHONY: test-upload
test-upload:
	@echo "🧪 Testing upload endpoint..."
	@echo "Note: This will fail with empty file (expected for testing)"
	curl -X POST "https://goofyahead--upload-audio.modal.run" \
		-H "Authorization: Bearer $(API_KEY)" \
		-F "audio_file=@/dev/null" -v

# Initialize database
.PHONY: setup-db
setup-db:
	@echo "🗄️ Initializing database..."
	curl -X POST "https://goofyahead--setup.modal.run" \
		-H "Authorization: Bearer $(API_KEY)"

# Kill running processes
.PHONY: clean
clean:
	@echo "🧹 Killing Streamlit processes..."
	-pkill -f streamlit
	@echo "✅ Clean complete"

# Install dependencies
.PHONY: install
install:
	@echo "📦 Installing Python dependencies..."
	pip install streamlit requests pandas plotly

# Development server with auto-reload
.PHONY: dev
dev:
	@echo "🔄 Starting development server with auto-reload..."
	API_AUTH_KEY=$(API_KEY) streamlit run streamlit_ui_enhanced.py --server.port $(STREAMLIT_PORT) --server.runOnSave true

# Run in background
.PHONY: run-bg
run-bg:
	@echo "🚀 Starting Streamlit UI in background..."
	API_AUTH_KEY=$(API_KEY) nohup streamlit run streamlit_ui_enhanced.py --server.port $(STREAMLIT_PORT) > streamlit.log 2>&1 &
	@echo "✅ Streamlit running in background (PID: $$!)"
	@echo "📋 Logs: tail -f streamlit.log"

# Show status
.PHONY: status
status:
	@echo "📊 Service Status:"
	@echo "Modal Health:"
	@curl -s -H "Authorization: Bearer $(API_KEY)" https://goofyahead--health.modal.run | python3 -m json.tool
	@echo ""
	@echo "Streamlit Processes:"
	@pgrep -f streamlit || echo "No Streamlit processes running"

# Test all endpoints
.PHONY: test-all
test-all:
	@echo "🧪 Testing all endpoints..."
	@echo "1. Health endpoint:"
	@curl -s -H "Authorization: Bearer $(API_KEY)" https://goofyahead--health.modal.run
	@echo ""
	@echo "2. Business Intelligence endpoint:"
	@curl -s -H "Authorization: Bearer $(API_KEY)" "https://goofyahead--business-intelligence.modal.run?action=deals_by_stage"
	@echo ""

# Open browser
.PHONY: open
open:
	@echo "🌐 Opening browser..."
	@python3 -c "import webbrowser; webbrowser.open('http://localhost:$(STREAMLIT_PORT)')"