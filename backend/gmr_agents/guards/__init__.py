"""Application-level guards the LangGraph agent passes every tool request through.

Who controls them: application code. The LLM never sees or bypasses these -- it
only receives their structured verdicts back as errors to correct.
"""
