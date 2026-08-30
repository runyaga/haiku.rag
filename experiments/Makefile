PY := $(CURDIR)/.venv/bin/python
CFG := HAIKU_RAG_CONFIG_PATH=$(CURDIR)/synth/config-synth.yaml PYTHONPATH=$(CURDIR)/src

.PHONY: check lint test corpus suites all clean
check: lint test          ## ruff + unit tests + 100% coverage gate
lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests
test:
	.venv/bin/python -m pytest --cov=fusionlab --cov-report=term-missing
corpus:                   ## build the three synthetic databases (needs the embedder)
	$(CFG) $(PY) -m fusionlab.build
suites:                   ## run the integration suites against the built corpora
	$(CFG) $(PY) -m pytest -m integration --no-cov
all: check corpus suites
clean:
	rm -rf synth/*.lancedb synth/truth.json synth/manifest.json .coverage .pytest_cache
