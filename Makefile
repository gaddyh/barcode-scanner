.PHONY: install run scan test bench lint docker-build docker-run

install:
	python -m pip install -e ".[dev]"

run:
	uvicorn app.main:app --reload

scan:
	barcode-scan $(IMAGE)

test:
	pytest

bench:
	python -m tests.benchmark.runner

lint:
	ruff check .

docker-build:
	docker build -t barcode-scanner .

docker-run:
	docker run --rm -p 8000:8000 barcode-scanner
