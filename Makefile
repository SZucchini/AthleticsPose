.PHONY: setup venv download data checkpoints clean

setup: venv download

venv:
	uv sync
	uv pip install -e .

download: data checkpoints

data:
	@set -e; \
	if command -v curl >/dev/null 2>&1; then \
		curl -L -o data.zip "https://github.com/SZucchini/AthleticsPose/releases/latest/download/data.zip"; \
	elif command -v wget >/dev/null 2>&1; then \
		wget -O data.zip "https://github.com/SZucchini/AthleticsPose/releases/latest/download/data.zip"; \
	else \
		echo "Error: need 'curl' or 'wget'." >&2; exit 1; \
	fi; \
	unzip -o data.zip; \
	rm -f data.zip

checkpoints:
	@set -e; \
	if command -v curl >/dev/null 2>&1; then \
		curl -L -o checkpoints.zip "https://github.com/SZucchini/AthleticsPose/releases/latest/download/checkpoints.zip"; \
	elif command -v wget >/dev/null 2>&1; then \
		wget -O checkpoints.zip "https://github.com/SZucchini/AthleticsPose/releases/latest/download/checkpoints.zip"; \
	else \
		echo "Error: need 'curl' or 'wget'." >&2; exit 1; \
	fi; \
	unzip -o checkpoints.zip; \
	rm -f checkpoints.zip

clean:
	rm -rf .venv
