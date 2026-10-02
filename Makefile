install-dev:
	@echo "Installing local package..."
	uv sync
	uv run prek install
lint:
	@echo "Running linters..."
	uv run ruff check
	uv run ruff format --check
	uv run mypy .
test:
	@echo "Running tests..."
	uv run pytest --cov=src/pydantic_ai_linkup/ --cov-report term-missing tests

update-dependencies:
	uv lock --upgrade
update-pre-commit-hooks:
	uv run prek autoupdate --cooldown-days 14
