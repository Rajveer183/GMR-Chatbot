"""Questions that span both systems: WFM (installation, SAT, O&M, stock, revenue) and MDM (HES meter data, power
quality, losses, reliability, theft, mapping audit).

Triage picks ONE domain, so 'do zones with more non-comm O&M tickets show more meters stopped communicating in MDM?'
was planned on the WFM catalog only and refused ('Unknown table communication_failures'). Here the SQL writer adds the
other system's best topic and plans on both catalogs; each query still runs on one engine (the SQL guard enforces it)
and the other system's figures come from a parallel extra query, grouped by the same place key so the answer -- and
the statistics in presentation/stats.py -- can line them up.
"""

from __future__ import annotations

import re

from gmr_agents.catalog import business_rules

# Words that only make sense in one system. Kept narrow: 'supply' (power supply), 'installed' (HES install date)
# and 'meters' alone appear in both, so they do not count.
_MDM_RE = re.compile(r"\bmdm\b|\bhes\b|meter data|stopped communicat|communication failure|feeder loss|\blt loss|"
                     r"\bdt loss|energy loss|energy audit|voltage|power factor|\bsaidi\b|\bsaifi\b|\bcaidi\b|reliability|"
                     r"outage|theft|load violation|dt load|overload|md breach|mapping audit|auto.?index|reverse flow|"
                     r"phase imbalance|current unbalance|consumption spike|hidden outage|data quality|data completeness", re.I)
_WFM_RE = re.compile(r"\bwfm\b|install|\bsat\b|o&m|\bom\b|ticket|complaint|stock|inventory|technician|defect|"
                     r"non.?comm|revenue|invoice|receivable|collection|journey|warehouse", re.I)

PLAN_HINT = (
    "CROSS-SYSTEM QUESTION: WFM tables (public.* on Postgres, gmr_wfm_silver.* on Athena) and MDM tables (hes_gold.* "
    "on Athena) are different systems -- never JOIN them. Put one system's figures in the main SQL and the other's in "
    "extra_queries, each grouped by the SAME key (project / zone / circle, UPPER(name)) and each following its own "
    "topic rules (MDM: one month, the dashboard default when none is asked). The answer lines them up by that key.")


def cross_domain(question: str, triage: dict) -> str | None:
    """The other domain ('wfm' / 'mdm') when the question needs both systems, else None."""
    if triage.get("source") == "router" or triage.get("kind") not in (None, "data"):
        return None
    q = str(question or "")
    mdm, wfm = bool(_MDM_RE.search(q)), bool(_WFM_RE.search(q))
    if not (mdm and wfm):
        return None
    other = "mdm" if triage.get("domain") == "wfm" else "wfm"
    topics = business_rules.topics()
    if any(topics.get(t, {}).get("domain", "wfm") == other for t in triage.get("topics") or []):
        return None   # already has a topic of the other system
    return other


def widen(question: str, triage: dict, other: str) -> dict:
    """Triage with the other system's best topic and its tables added (own best topic kept first)."""
    topics = business_rules.topics()
    ranked = [t for t in business_rules.rank_topics(question, len(topics))
              if t in topics and topics[t].get("domain", "wfm") == other]
    if not ranked:
        return triage
    own = list(triage.get("topics") or [])[:1]
    extra_tables = business_rules.tables_for(ranked[:1])
    return {**triage, "topics": [*own, ranked[0], *list(triage.get("topics") or [])[1:]][:3],
            "tables": list(dict.fromkeys([*(triage.get("tables") or []), *extra_tables])),
            "cross_domain": other}
