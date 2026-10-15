"""Read-only connectors to the GMR DISCOM data lake: Athena (athena.py) and the
WFM gold Postgres (postgres.py). Settings come from backend/.env (config.py);
readonly.py rejects anything but a single SELECT before it is sent.
"""
