.PHONY: setup test server dashboard demo https verify docker-build docker-up docker-down clean lfs-pull

PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PY := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

SERVER_HOST ?= 0.0.0.0
SERVER_PORT ?= 8000
DASHBOARD_PORT ?= 8501

# ------------------------------------------------------------------------
# setup: create/refresh the virtualenv and install project dependencies.
# ------------------------------------------------------------------------
setup:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r case1/requirements.txt
	@echo ""
	@echo "Setup complete. Activate with: source $(VENV_DIR)/bin/activate"
	@echo "If model weights show up as tiny text pointer files, run: make lfs-pull"

# Fetch real Git LFS binaries (model weights, .pt/.torchscript/images) if
# only pointer files are checked out (e.g. after a shallow/no-lfs clone).
lfs-pull:
	git lfs pull

# ------------------------------------------------------------------------
# test: run the full case1 pytest suite. Data-/weight-dependent tests are
# skipped automatically (see case1/tests/conftest.py and pytest.mark.skipif
# guards) when the required local artifacts are not present.
# ------------------------------------------------------------------------
test:
	$(VENV_PY) -m pytest case1/tests -q

# ------------------------------------------------------------------------
# verify: run the adversarial GeoJSON / ISO-XML TaskData verification script.
# ------------------------------------------------------------------------
verify:
	$(VENV_PY) scripts/verify_prescription_and_taskdata.py

# ------------------------------------------------------------------------
# server / dashboard: run each service standalone (foreground).
# ------------------------------------------------------------------------
server:
	$(VENV_PY) -m uvicorn case1.server.app:app --host $(SERVER_HOST) --port $(SERVER_PORT) --reload

dashboard:
	$(VENV_PY) -m streamlit run case1/dashboard/app.py --server.port $(DASHBOARD_PORT)

# ------------------------------------------------------------------------
# demo: run the FastAPI server and Streamlit dashboard together.
# Ctrl+C stops both (the trap kills the background server on exit).
# ------------------------------------------------------------------------
demo:
	@trap 'kill %1 2>/dev/null' EXIT; \
	$(VENV_PY) -m uvicorn case1.server.app:app --host $(SERVER_HOST) --port $(SERVER_PORT) & \
	sleep 2; \
	$(VENV_PY) -m streamlit run case1/dashboard/app.py --server.port $(DASHBOARD_PORT)

# ------------------------------------------------------------------------
# https: serve the FastAPI app (and therefore the /mobile PWA) over HTTPS
# so the Service Worker can register from a phone on the local network.
# See scripts/serve_https.sh for mkcert / tunnel options.
# ------------------------------------------------------------------------
https:
	bash scripts/serve_https.sh

# ------------------------------------------------------------------------
# Docker
# ------------------------------------------------------------------------
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} +
	find . -name "*.pyc" -not -path "./.git/*" -delete
	rm -rf .pytest_cache
