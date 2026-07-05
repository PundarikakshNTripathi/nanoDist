.PHONY: setup lint format test docker-up docker-down

setup:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run mypy src/

format:
	uv run ruff format .

test:
	uv run pytest tests/ --cov=src/distributed_trainer

docker-up:
	cd docker && docker-compose up --build -d

docker-down:
	cd docker && docker-compose down
