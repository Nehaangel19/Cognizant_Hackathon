# Predictive Maintenance Agent — Makefile
# Run with: make <target>
# Works in Git Bash on Windows. No PowerShell-only syntax.

.PHONY: setup train verify test calibrate demo cache all clean help

help:
	@echo "Targets:"
	@echo "  make setup     - Create venv and install dependencies"
	@echo "  make train     - Train all models (~30s)"
	@echo "  make verify    - Validate rules + verify all claims"
	@echo "  make test      - Run the regression test suite"
	@echo "  make calibrate - Fit probability calibration"
	@echo "  make demo      - Run the pre-seeded demo cases (offline)"
	@echo "  make cache     - Build precomputed demo data cache"
	@echo "  make all       - setup -> train -> calibrate -> verify -> test"
	@echo "  make clean     - Remove model artefacts and cache files"

setup:
	python -m venv venv
	./venv/Scripts/activate && pip install -r requirements.txt && pip install pytest

train:
	./venv/Scripts/python.exe src/train.py

calibrate:
	./venv/Scripts/python.exe src/calibration.py

verify:
	./venv/Scripts/python.exe src/validate_rules.py
	./venv/Scripts/python.exe scripts/verify_claims.py

test:
	./venv/Scripts/python.exe -m pytest tests/ -v

demo:
	./venv/Scripts/python.exe src/agent/agent.py --demo

cache:
	./venv/Scripts/python.exe scripts/build_demo_cache.py

all: setup train calibrate verify test
	@echo ""
	@echo "All tasks complete. Run 'make demo' to see the agent in action."

clean:
	rm -rf models/*.joblib
	rm -f data/demo_cache.json data/work_orders.json
	rm -f docs/calibration_curve.png docs/verification_report.md
	rm -rf .pytest_cache tests/__pycache__
