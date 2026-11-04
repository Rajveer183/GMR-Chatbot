"""The LLM agents -- one LLM call each (plus a bounded repair):

    router/      Intent Router + Question Refiner   (triage; skipped for verified wordings)
    sql_writer/  SQL Writer + Query Debugger        (plan, validate, execute, repair)
    narrator/    Answer Narrator + Next-Question Suggester + Fact Checker
    knowledge/   Glossary Keeper / Doc Librarian    (RAG explain answers)

Everything else (retrieval, guards, checks, execution, charts) is deterministic code."""
