#!/bin/sh
cd "$(dirname "$0")"
exec .venv/bin/streamlit run src/marl_path/visualization/app.py --server.headless true --server.port 8501
