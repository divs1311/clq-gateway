.PHONY: install data eval eval-live simulate test dashboard all

install:
	pip install -e ".[dev]"

data:
	python scripts/build_dataset.py

eval:
	python -m clq.cli eval --mode mock

eval-live:   ## needs CLQ_SMALL_MODEL, CLQ_MID_MODEL, CLQ_FRONTIER_MODEL, CLQ_JUDGE_MODEL + provider keys
	python -m clq.cli eval --mode litellm

simulate:
	python -m clq.cli simulate

test:
	python -m pytest -q

dashboard: eval
	python scripts/build_dashboard.py

all: data eval test dashboard
