"""
landscape_agent.py — Multi-agent literature landscape builder.

This module implements a pipeline of specialised agents that together produce
a structured literature landscape from a user's research interest. Each agent
has a clearly defined input, output, and responsibility.

Pipeline overview:
    0. anchor_builder   — Parses user input into a structured anchor document
                          that defines the scope for all downstream agents.
    1. literature_scout — Fetches candidate papers from PubMed using keyword
                          search, citation traversal, and author network tools
                          (stub; full implementation in Session 2).
    2. relevance_ranker — Scores each candidate paper 0-100 against the anchor
                          and returns them sorted by relevance
                          (stub; full implementation in Session 2).
    3. synthesis_agent  — Generates the final landscape report from the
                          anchor and selected papers
                          (stub; full implementation in Session 3).

Relationship to other modules:
    Imports get_pmcid() and fetch_abstract() from research_agent.py so the
    landscape pipeline can check full-text availability and retrieve abstracts
    without duplicating that logic.
"""

import os
import re
import json
from anthropic import Anthropic
from concurrent.futures import ThreadPoolExecutor, as_completed
from research_agent import get_pmcid, fetch_abstract

# Initialise the Anthropic client once at module level. The client reads
# ANTHROPIC_API_KEY from the environment; load_dotenv() in research_agent.py
# will have already populated os.environ by the time this module is imported.
client = Anthropic()


def anchor_builder(user_input: str) -> dict:
    """
    Agent 0: Analyses user input and builds a structured anchor document that
    defines the scope for the entire literature landscape pipeline.

    Uses Claude to propose a research anchor — the organising frame that all
    subsequent agents (scout, ranker, synthesiser) will use to filter and
    contextualise papers. The anchor captures the central question, field
    boundaries, key intellectual tensions, and search strategy flags.

    PMID detection: if the user's input contains a 7-8 digit number, it is
    assumed to be a PubMed ID and citation_traversal is enabled so the scout
    can fetch citing and cited papers around that seed paper.

    Args:
        user_input (str): Free text from the user — may be a topic, a research
                          question, a PMID with context, or any combination.

    Returns:
        dict with keys:
            core_question (str):      Central research question the landscape
                                      should answer, specific and literature-
                                      answerable.
            field_scope (str):        Field, subfield, time range, and key
                                      boundaries for the search.
            key_debates (list[str]):  3-5 genuine intellectual tensions or
                                      methodological debates in this field.
            search_strategy (dict):   Boolean flags controlling which retrieval
                                      tools the scout will activate:
                                          keyword_search (bool):     always True
                                          citation_traversal (bool): True if PMID found
                                          author_network (bool):     True by default
            detected_pmid (str|None): The PMID extracted from the input, or None
                                      if no PMID was found.
    """
    # ── PMID detection ────────────────────────────────────────────────────────
    # PMIDs are 7-8 digit numbers. \b word boundaries prevent matching digit
    # sequences that are part of longer numbers (e.g. years, DOIs, version strings).
    pmid_match = re.search(r'\b\d{7,8}\b', user_input)
    detected_pmid = pmid_match.group() if pmid_match else None

    # Append PMID context to the prompt so Claude sets citation_traversal
    # correctly in the returned JSON. Without this hint, Claude has no reliable
    # way to know a PMID was detected by the regex above.
    pmid_context = ""
    if detected_pmid:
        pmid_context = (
            f"\n\nNote: The user's input contains PMID {detected_pmid}. "
            "Set citation_traversal to true."
        )

    # ── Claude API call ───────────────────────────────────────────────────────
    # claude-sonnet-4-6: reliable instruction-following for constrained JSON
    # output; strong enough to identify real debates in a field from a short
    # user prompt without the latency of Opus.
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        system=(
            "You are a research scoping assistant. "
            "Analyse the user's research interest and return a structured "
            "anchor document as valid JSON only. "
            "No markdown, no backticks, no preamble. Just the JSON object."
        ),
        messages=[{
            "role": "user",
            "content": (
                f'Analyse this research interest and return a JSON anchor document:\n\n'
                f'"{user_input}"{pmid_context}\n\n'
                "Return ONLY this JSON structure:\n"
                "{\n"
                '    "core_question": "One precise research question this landscape should answer",\n'
                '    "field_scope": "The field, subfield, time range and key boundaries for this search",\n'
                '    "key_debates": [\n'
                '        "Debate or tension 1 in this field",\n'
                '        "Debate or tension 2",\n'
                '        "Debate or tension 3",\n'
                '        "Debate or tension 4"\n'
                "    ],\n"
                '    "search_strategy": {\n'
                '        "keyword_search": true,\n'
                '        "citation_traversal": false,\n'
                '        "author_network": true\n'
                "    }\n"
                "}\n\n"
                "Make the core_question specific and answerable from the literature.\n"
                "Make the field_scope include a time range (e.g. \"2010-present\").\n"
                "Key debates should reflect genuine tensions in the field.\n"
                "Set citation_traversal to true only if a PMID was detected."
            ),
        }],
    )

    # ── Parse the response ────────────────────────────────────────────────────
    raw = next(
        (b.text for b in response.content if b.type == "text"),
        "{}",
    )

    # Strip accidental markdown fences if Claude added them despite the system
    # prompt instruction. Split on the first newline after the opening fence,
    # then strip the closing fence from the end.
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    try:
        anchor = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback anchor if JSON parsing fails — ensures downstream agents
        # always receive a well-structured dict even if the API response is
        # malformed or truncated.
        anchor = {
            "core_question": user_input,
            "field_scope": "Biomedical literature, 2010-present",
            "key_debates": [
                "Methods comparison",
                "Mechanistic vs correlative approaches",
            ],
            "search_strategy": {
                "keyword_search": True,
                "citation_traversal": detected_pmid is not None,
                "author_network": True,
            },
        }

    # Always attach detected_pmid so downstream agents can use it directly
    # without re-running the regex on the original user input.
    anchor["detected_pmid"] = detected_pmid

    # Override citation_traversal to ensure it is always consistent with PMID
    # detection — Claude may set it incorrectly in edge cases.
    if detected_pmid:
        anchor["search_strategy"]["citation_traversal"] = True

    return anchor


def literature_scout(anchor: dict) -> list:
    """
    Agent 1: Fetches candidate papers from PubMed based on the anchor document.

    Uses three retrieval tools in parallel based on the anchor's search_strategy
    flags:
      - keyword_search:     PubMed esearch on terms derived from core_question
      - citation_traversal: fetch papers citing and cited by the seed PMID
      - author_network:     fetch recent papers by key authors in the field

    For each retrieved paper, checks full-text availability via get_pmcid() and
    fetches the abstract via fetch_abstract() to support relevance ranking.

    Args:
        anchor (dict): Anchor document produced by anchor_builder(), containing
                       core_question, field_scope, search_strategy, and
                       optionally detected_pmid.

    Returns:
        list[dict]: Candidate papers. Each dict has keys:
            pmid (str):           PubMed ID
            title (str):          Paper title
            authors (str):        Author list
            journal (str):        Journal name
            year (str):           Publication year
            relationship (str):   How this paper was found
                                  ("keyword", "cited_by", "cites", "author_network")
            has_full_text (bool): True if a PMCID exists in PMC
            pmcid (str|None):     PMCID if available, else None

    NOTE: Stub implementation — returns an empty list.
    Full implementation planned for Session 2.
    """
    # TODO: Session 2 — implement keyword_search, citation_traversal, author_network
    # tools and run them in parallel with ThreadPoolExecutor. Use get_pmcid() and
    # fetch_abstract() from research_agent.py for each retrieved PMID.
    return []


def relevance_ranker(candidates: list, anchor: dict) -> list:
    """
    Agent 2: Scores and ranks candidate papers by relevance to the anchor.

    Sends all candidates and the anchor document to Claude in a single API call.
    Claude scores each paper 0-100 against the anchor's core_question and
    field_scope, and returns a one-sentence rationale for each score.

    Scoring rubric (intended for Session 2 implementation):
      90-100: Directly answers the core question; canonical reference.
      70-89:  Highly relevant; covers key methods or debates in scope.
      50-69:  Peripherally relevant; useful context but not central.
      <50:    Off-topic or outside field_scope time range.

    Args:
        candidates (list[dict]): Papers returned by literature_scout(), each
                                 with pmid, title, authors, journal, year, and
                                 relationship keys.
        anchor (dict):           Anchor document for scoring context —
                                 core_question and field_scope are used as
                                 the primary scoring criteria.

    Returns:
        list[dict]: Same papers sorted by relevance_score descending, with two
                    keys added to each dict:
                        relevance_score (int):      0-100
                        relevance_rationale (str):  one-sentence explanation

    NOTE: Stub implementation — returns candidates unchanged (no scoring).
    Full implementation planned for Session 2.
    """
    # TODO: Session 2 — build a single prompt with all candidates and the anchor,
    # parse Claude's scored JSON response, attach scores to each dict, and sort
    # descending by relevance_score.
    return candidates


def synthesis_agent(anchor: dict, selected_papers: list) -> str:
    """
    Agent 3: Generates the structured literature landscape report.

    Uses the anchor document as the organising frame and synthesises the
    selected papers into a six-section markdown report:

      1. Context        — Why this question matters; state of the field.
      2. Contributions  — What the selected papers collectively establish.
      3. Built upon     — Foundational work these papers cite and extend.
      4. Parallel work  — Contemporary approaches tackling the same question.
      5. Open questions — Gaps, contradictions, and unresolved debates.
      6. Reading order  — Recommended sequence for a newcomer to the field.

    The anchor's key_debates drive the "Open questions" section; the anchor's
    core_question frames the "Context" section.

    Args:
        anchor (dict):              Confirmed anchor document from anchor_builder(),
                                    containing core_question, field_scope, and
                                    key_debates.
        selected_papers (list[dict]): User-curated final paper list. Each dict
                                      should include pmid, title, authors, year,
                                      and a summary text field attached by the
                                      pipeline before this agent is called.

    Returns:
        str: Structured markdown landscape report ready for display in the UI
             or export as a document.

    NOTE: Stub implementation — returns a placeholder string.
    Full implementation planned for Session 3.
    """
    # TODO: Session 3 — build a prompt that includes the anchor document and
    # all selected paper summaries, then ask Claude to produce the six-section
    # landscape report using the anchor's core_question and key_debates as
    # the organising frame.
    return "## Literature landscape\n\nSynthesis coming in Session 3."
