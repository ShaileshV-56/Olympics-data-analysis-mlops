"""Ensure required packages exist in requirements.txt.

This script will add `streamlit` to requirements.txt if it's missing.
It leaves existing versions intact and appends a minimal `streamlit>=1.0` entry when absent.
"""
from pathlib import Path

req_path = Path(__file__).parent.parent / 'requirements.txt'

if not req_path.exists():
    print(f"requirements.txt not found at {req_path}. Creating a new one.")
    req_path.write_text('')

lines = [l.rstrip('\n') for l in req_path.read_text().splitlines()]
lines_lower = [l.strip().lower() for l in lines if l.strip() and not l.strip().startswith('#')]

has_streamlit = any('streamlit' in l for l in lines_lower)

if has_streamlit:
    print('streamlit already present in requirements.txt')
else:
    print('streamlit not found in requirements.txt — adding streamlit>=1.0')
    # Append with minimal version
    lines.append('streamlit>=1.0')
    req_path.write_text('\n'.join(lines) + '\n')
    print('requirements.txt updated')

print('Done')
