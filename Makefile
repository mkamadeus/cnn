test:
	python -m unittest tests/test_*.py -v
.PHONY: test

format:
	black ./**/*.py --exclude env/
.PHONY: format

lint:
	flake8 .
	black ./**/*.py --check --exclude env/
.PHONY: lint
