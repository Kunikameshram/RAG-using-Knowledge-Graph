# RAG using Knowledge Graph

This project demonstrates Retrieval-Augmented Generation (RAG) using a Knowledge Graph built with Neo4j and enhanced with LLM (Large Language Model) capabilities via LangChain and OpenAI.

## Features

- **Document Loading:** Fetches and splits documents (e.g., from Wikipedia) for processing.
- **Knowledge Graph Construction:** Uses LLMs to extract entities and relationships, then builds a graph in Neo4j.
- **RAG Pipeline:** Enables advanced question answering and retrieval by leveraging both the knowledge graph and LLMs.

## Setup

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd RAG\ using\ Knowledge\ Graph
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   - Create a `.env` file in the root directory with the following content:
     ```
     OPENAI_API_KEY=your-openai-api-key
     NEO4J_URI=neo4j+s://<your-neo4j-uri>
     NEO4J_USERNAME=neo4j
     NEO4J_PASSWORD=your-neo4j-password
     ```

## Usage

1. **Run the main pipeline:**
   ```sh
   python3 -m src.main
   ```

2. **Test Neo4j connection (optional):**
   ```sh
   python3 -m src.neo4j_test
   ```

## Requirements

- Python 3.8+
- [Neo4j Aura](https://console.neo4j.io/) (or another Neo4j instance)
- OpenAI API key
