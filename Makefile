.PHONY: test lint format run-simulation

test:
	pytest tests/

lint:
	ruff check .
	mypy src/

format:
	ruff format .

run-simulation:
	python scripts/run_hpc_simulation.py

docker-up:
	cd docker && docker-compose up -d

docker-down:
	cd docker && docker-compose down
