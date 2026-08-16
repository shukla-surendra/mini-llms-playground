PYTHON := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
# Cross-platform (macOS/Linux) LAN IP lookup: opens a UDP "connection" (no packets actually
# sent) to a public IP just to ask the OS which local interface/address it would route
# through - avoids branching on ifconfig (mac/BSD) vs ip (linux) output formats.
LAN_IP := $(shell $(PYTHON) -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()" 2>/dev/null)

.PHONY: docs check

docs: ## Serve the MkDocs Material site at http://0.0.0.0:8000 (reachable from other machines on the network; live-reloads on edits to docs/*.md)
	@echo "$(LAN_IP)"
	@echo "Serving at http://localhost:8000  (on the network: http://$(LAN_IP):8000)"
	$(PYTHON) -m mkdocs serve -a 0.0.0.0:8000

check: ## Validate all relative Markdown links repo-wide, then build the MkDocs site (no serve) - CI-friendly
	$(PYTHON) scripts/check_links.py
	$(PYTHON) -m mkdocs build
