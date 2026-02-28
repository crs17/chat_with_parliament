# Chat med Folketinget

The purpose of this repo is to demonstrate an implementation of *AI agent delegation*. Specifically, we will demonstrate agent delegation with a political analyst agent directing a series of party expert agents.

The demo can be viewed [here](https://crs17.github.io/chat_with_parliament/).

## Requirements
- uv
- Ollama local installation
- Docker
- logfire account for debugging

Set up `.env` and adjust if needed: `cp .env.example .env`.

## Tech Stack overview
- Playwright for fetching party manifests on party sites that are defensive against automated requests
- Docling for for parsing HTML files into markdown
- Chonkie for chunking
- Local Ollama deployment for LLM and embedding models. We do not use docker here as GPU access from docker is not available on Apple Silicon.
- Weaviate for vector database in docker
- Pydantic AI for Agent orchestration
- Logfire for monitoring and debugging of the Agents
