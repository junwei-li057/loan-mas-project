"""Compatibility entry point for the single-page Streamlit demo.

The maintained demo lives in app/ui/Home.py. This wrapper keeps the old
command working:

    streamlit run app/loan_demo.py
"""
from __future__ import annotations

import runpy
from pathlib import Path


HOME_PATH = Path(__file__).resolve().parent / "ui" / "Home.py"
runpy.run_path(str(HOME_PATH), run_name="__main__")
