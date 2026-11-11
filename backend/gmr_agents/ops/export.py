"""Export a whole chat session as a report: GET /api/sessions/{id}/export?format=md|html.

Built from the answers already stored in the session (question, summary, key metrics,
tables, insights, warnings and -- optionally -- the SQL), so exporting never re-runs a query
or calls an LLM. Tables are capped at MAX_ROWS rows each; the full table is still one click
away in the app (paged download).
"""

from __future__ import annotations

import html
import json
import re

MAX_ROWS = 50


def _rows(table: dict) -> tuple[list[str], list[list[str]]]:
    cols = [str(c) for c in table.get("columns") or []]
    rows = []
    for r in (table.get("rows") or [])[:MAX_ROWS]:
        if isinstance(r, dict):
            rows.append([str(r.get(c, "")) for c in cols])
        else:
            rows.append([str(v) for v in r])
    return cols, rows


def _md_cell(v: str) -> str:
    return v.replace("|", "\\|").replace("\n", " ")


def _turns(session: dict) -> list[dict]:
    """Pair each question with the answer that followed it."""
    turns, pending = [], None
    for m in session.get("messages") or []:
        if m.get("role") == "user":
            pending = {"question": m.get("content") or "", "at": m.get("timestamp"), "response": None}
            turns.append(pending)
        elif m.get("role") == "assistant" and m.get("response") is not None:
            if pending is None or pending["response"] is not None:
                pending = {"question": (m["response"] or {}).get("question") or "", "at": m.get("timestamp"), "response": None}
                turns.append(pending)
            pending["response"] = m["response"]
    return turns


def to_markdown(session: dict, include_sql: bool = False) -> str:
    out = [f"# {session.get('session_name') or 'Chat session'}", "",
           f"_Exported conversation -- {session.get('created_at', '')} to {session.get('updated_at', '')}_", ""]
    for i, t in enumerate(_turns(session), 1):
        r = t["response"] or {}
        out += [f"## {i}. {t['question']}", ""]
        if r.get("summary"):
            out += [r["summary"], ""]
        kpis = r.get("kpis") or []
        if kpis:
            out += [f"- **{k.get('label')}**: {k.get('formatted_value') or k.get('value')}" for k in kpis] + [""]
        for tb in r.get("tables") or []:
            cols, rows = _rows(tb)
            if not cols:
                continue
            if tb.get("title"):
                out += [f"**{tb['title']}**", ""]
            out += ["| " + " | ".join(map(_md_cell, cols)) + " |", "|" + "---|" * len(cols)]
            out += ["| " + " | ".join(map(_md_cell, row)) + " |" for row in rows]
            total = tb.get("total_rows")
            if isinstance(total, int) and total > len(rows):
                out.append(f"\n_{len(rows)} of {total} rows shown._")
            out.append("")
        if r.get("insights"):
            out += ["**Insights**", ""] + [f"- {x}" for x in r["insights"]] + [""]
        if r.get("warnings"):
            out += ["**Notes**", ""] + [f"- {x}" for x in r["warnings"]] + [""]
        sql = (r.get("debug") or {}).get("sql")
        if include_sql and sql:
            out += ["```sql", str(sql), "```", ""]
    return "\n".join(out).rstrip() + "\n"


def _inline(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(text))


def to_html(session: dict, include_sql: bool = False) -> str:
    """A self-contained printable page (print -> PDF from the browser)."""
    title = html.escape(session.get("session_name") or "Chat session")
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>"
             "body{font:14px/1.5 system-ui,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#1d2433;background:#fff}"
             "h1{font-size:22px}h2{font-size:16px;margin-top:28px;border-top:1px solid #e3e6ec;padding-top:16px}"
             "table{border-collapse:collapse;margin:8px 0;font-size:13px}td,th{border:1px solid #d5d9e0;padding:4px 8px;text-align:left}"
             "th{background:#f3f5f8}.kpi{display:inline-block;margin:4px 12px 4px 0}.muted{color:#667085;font-size:12px}"
             "pre{background:#f6f8fa;padding:8px;overflow:auto;font-size:12px}</style></head><body>",
             f"<h1>{title}</h1><p class='muted'>Exported conversation -- {html.escape(str(session.get('created_at', '')))}"
             f" to {html.escape(str(session.get('updated_at', '')))}</p>"]
    for i, t in enumerate(_turns(session), 1):
        r = t["response"] or {}
        parts.append(f"<h2>{i}. {html.escape(t['question'])}</h2>")
        if r.get("summary"):
            parts.append(f"<p>{_inline(r['summary'])}</p>")
        for k in r.get("kpis") or []:
            parts.append(f"<span class='kpi'><strong>{html.escape(str(k.get('label')))}</strong>: "
                         f"{html.escape(str(k.get('formatted_value') or k.get('value')))}</span>")
        for tb in r.get("tables") or []:
            cols, rows = _rows(tb)
            if not cols:
                continue
            if tb.get("title"):
                parts.append(f"<p><strong>{html.escape(tb['title'])}</strong></p>")
            parts.append("<table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr>"
                         + "".join("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in row) + "</tr>" for row in rows)
                         + "</table>")
        for label, key in (("Insights", "insights"), ("Notes", "warnings")):
            if r.get(key):
                parts.append(f"<p><strong>{label}</strong></p><ul>" + "".join(f"<li>{_inline(str(x))}</li>" for x in r[key]) + "</ul>")
        sql = (r.get("debug") or {}).get("sql")
        if include_sql and sql:
            parts.append(f"<pre>{html.escape(str(sql))}</pre>")
    parts.append("</body></html>")
    return "".join(parts)


def to_json(session: dict) -> str:
    return json.dumps(session, ensure_ascii=False, indent=1, default=str)
