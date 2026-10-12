.PHONY: help install install-ml migrate api ui test lint fmt docker clean

help:
	@echo "install     Install runtime + test dependencies"
	@echo "install-ml  Add the optional ML extras (Whisper, sentence-transformers)"
	@echo "migrate     Apply database migrations"
	@echo "api         Run the API with autoreload on :8000"
	@echo "ui          Run the Streamlit UI on :8501"
	@echo "test        Run the test suite"
	@echo "lint        Ruff check"
	@echo "fmt         Ruff format + autofix"
	@echo "docker      Build and start the full stack"

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

install-ml:
	pip install -r requirements-ml.txt

migrate:
	alembic upgrade head

api:
	uvicorn app.main:app --reload --port 8000

ui:
	streamlit run frontend/streamlit_app.py

test:
	pytest -q

lint:
	ruff check app frontend tests

fmt:
	ruff check --fix app frontend tests
	ruff format app frontend tests

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache storage polyglot.db
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
