setup:
	uv sync
	uv run playwright install chromium

fetch_models:
	ollama pull qwen3:8b
	ollama pull nomic-embed-text-v2-moe:latest

logfire_auth:
	uv run logfire auth
	uv run logfire projects use

run:
	docker compose up -d

stop:
	docker compose down

clean:
	docker compose down
	docker volume rm weaviate_data

populate_db:
	uv run -m scripts.populate_db

# Quarto
quarto-preview:
	quarto preview chat_with_parliament.ipynb --no-browser --port 5080

quarto-render:
	quarto render chat_with_parliament.ipynb
	touch docs/.nojekyll