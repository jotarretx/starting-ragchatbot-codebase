# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Dependencies
```bash
# Install/update dependencies
uv sync

# Add new dependency
uv add package_name
```

### Environment Setup
**Prerequisites:** Install Ollama and download Mistral 7B model:
```bash
# Install Ollama (run with sudo)
curl -fsSL https://ollama.com/install.sh | sh

# Download Mistral 7B model
ollama pull mistral:7b
```

Optional `.env` configuration (defaults work for local setup):
```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b
```

## Architecture Overview

This is a **RAG (Retrieval-Augmented Generation) system** for course materials with a tool-calling architecture where the local AI model (Mistral 7B via Ollama) dynamically searches course content during response generation.

### Core Components

**RAG System (`rag_system.py`)** - Central orchestrator that coordinates all components
- Initializes and manages document processor, vector store, AI generator, session manager, and tool manager
- Handles query flow from API to AI generation with tool calling
- Manages conversation history and source tracking

**Document Processing Pipeline**
- `DocumentProcessor` parses structured course documents with metadata extraction
- Expects format: Course Title → Course Link → Course Instructor → Lesson sections
- Implements intelligent sentence-based chunking with configurable overlap
- Adds contextual prefixes to chunks for better retrieval

**Vector Storage (`vector_store.py`)**
- ChromaDB with sentence-transformers embeddings (`all-MiniLM-L6-v2`)
- Separate collections for course content and metadata
- Implements semantic search with distance filtering
- Handles both exact course matching and fuzzy name resolution

**Tool-Calling Architecture (`search_tools.py`)**
- `CourseSearchTool` exposes vector search as a function to the AI model
- AI decides when to search based on query context (not every query triggers search)
- Supports course name filtering and lesson-specific searches
- Source tracking for citation purposes

**AI Generation (`ai_generator.py`)**
- Ollama integration with Mistral 7B model and tool calling capabilities
- System prompt optimized for educational content responses
- Temperature 0 for consistent responses, 800 token limit
- Processes tool results during response generation

### Data Flow

1. **Document Ingestion**: Course files → Document processor → Vector chunks → ChromaDB
2. **Query Processing**: User query → RAG system → AI generator with tools
3. **Dynamic Search**: Mistral 7B decides to use search tool → Vector similarity search → Context injection
4. **Response Generation**: AI synthesizes search results → Structured response with sources

### Key Configuration (`config.py`)

- `CHUNK_SIZE`: 800 characters (sentence-based chunking)
- `CHUNK_OVERLAP`: 100 characters for context preservation  
- `MAX_RESULTS`: 5 search results per query
- `MAX_HISTORY`: 2 conversation turns for context
- `OLLAMA_MODEL`: mistral:7b (local model via Ollama)

### Frontend Integration

- Static HTML/CSS/JS served by FastAPI
- Real-time chat interface with loading states
- Markdown rendering for AI responses
- Collapsible source citations
- Session management for conversation continuity

### File Structure Context

- `/backend/` - All Python server code
- `/frontend/` - Static web assets (HTML/CSS/JS)  
- `/docs/` - Course material files (auto-loaded on startup)
- `run.sh` - Development server startup script

### Development Notes

- No test framework currently implemented
- No linting/formatting tools configured
- Uses `uv` for Python dependency management
- ChromaDB data persists in `./chroma_db/` 
- Application auto-loads documents from `/docs` on startup
- Frontend uses relative API paths (`/api/*`) for deployment flexibility
- Always use uv to run the server, do not use pip
- make sure to use uv to manage all dependencies
- Use uv to run python files