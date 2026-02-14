setup:
	uv sync

run:
	docker compose up -d

stop:
	docker compose down

clean:
	docker compose down
	docker volume rm weaviate_data

populate_db:
	uv run -m scripts.populate_db