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
import time
import requests
from anthropic import Anthropic
from concurrent.futures import ThreadPoolExecutor, as_completed
from research_agent import get_pmcid, fetch_abstract

# Initialise the Anthropic client once at module level. The client reads
# ANTHROPIC_API_KEY from the environment; load_dotenv() in research_agent.py
# will have already populated os.environ by the time this module is imported.
client = Anthropic()

# Semantic Scholar API base URL and optional key.
# Unauthenticated requests are rate-limited to ~100 req/5 min;
# an API key raises the limit substantially. Set SEMANTIC_SCHOLAR_API_KEY
# in .env to enable authenticated requests.
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")


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


def semantic_request(endpoint: str, params: dict) -> dict | None:
    """
    Makes a rate-limit-safe GET request to the Semantic Scholar API.

    Implements exponential backoff on HTTP 429 (Too Many Requests) responses,
    as required by the Semantic Scholar API terms of service. All other non-200
    status codes are treated as non-retryable and return None immediately.

    Backoff schedule: 1 s, 2 s, 4 s, 8 s — four attempts total before giving up.

    Args:
        endpoint (str): API path, e.g. "/paper/search" or
                        "/paper/PMID:12345678/citations".
        params (dict):  Query parameters forwarded to requests.get().

    Returns:
        dict: Parsed JSON response body on success, or None if all retries fail
              or a non-retryable error is encountered.
    """
    url = SEMANTIC_SCHOLAR_BASE + endpoint

    # Include the API key header when available; falls back to unauthenticated
    # (lower rate limit) if the environment variable is not set.
    headers = {}
    if SEMANTIC_SCHOLAR_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_KEY

    for attempt in range(5):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=10,
            )

            if response.status_code == 200:
                return response.json()

            elif response.status_code == 429:
                # Exponential backoff: 2**0=1s, 2**1=2s, 2**2=4s, 2**3=8s, 2**4=16s.
                # Five attempts gives the API up to 31 s of cumulative wait time,
                # which is sufficient for the unauthenticated /paper/search endpoint
                # to recover when called back-to-back with citation/reference lookups.
                wait_time = 2 ** attempt
                print(f"[rate limit] waiting {wait_time}s before retry")
                time.sleep(wait_time)

            else:
                # Non-retryable HTTP error — log and give up immediately.
                print(f"[semantic scholar] error {response.status_code}")
                return None

        except Exception as e:
            print(f"[semantic scholar] request failed: {e}")
            return None

    print("[semantic scholar] all retries exhausted")
    return None


def normalise_paper(
    pmid: str,
    title: str,
    authors: str,
    journal: str,
    year: int,
    relationship: str,
    intent: str = "",
) -> dict:
    """
    Creates a normalised paper dict used consistently throughout the pipeline.

    All retrieval tool functions return lists of these dicts so Agent 2
    (relevance_ranker) and Agent 3 (synthesis_agent) can compare and process
    papers from different sources without handling source-specific formats.

    Full-text availability is checked here by calling get_pmcid() from
    research_agent.py. This is a network call, so normalise_paper is
    intentionally called once per paper rather than in a tight loop.

    Args:
        pmid (str):         PubMed ID of the paper.
        title (str):        Full paper title.
        authors (str):      Formatted author string, e.g. "Teles et al."
        journal (str):      Journal name, or "" if unavailable.
        year (int):         Publication year, or 0 if unknown.
        relationship (str): How this paper was found relative to the anchor.
                            One of: "cites_anchor", "cited_by_anchor",
                            "same_author", "keyword_match".
        intent (str):       Semantic Scholar citation intent if available —
                            "background", "methodology", "result", or "".

    Returns:
        dict: Normalised paper record with keys:
              pmid, title, authors, journal, year, relationship,
              intent, has_full_text (bool), pmcid (str or None).
    """
    # get_pmcid returns None if the paper has no free full-text deposit in PMC.
    pmcid = get_pmcid(pmid)

    return {
        "pmid": pmid,
        "title": title,
        "authors": authors,
        "journal": journal,
        "year": year,
        "relationship": relationship,
        "intent": intent,
        "has_full_text": pmcid is not None,
        "pmcid": pmcid,
    }


def semantic_search(query: str, limit: int = 15) -> list[dict]:
    """
    Searches Semantic Scholar by keyword and semantic similarity.

    Unlike a PubMed keyword search, this endpoint uses dense vector similarity
    to find papers whose meaning matches the query, not just papers whose full
    text contains the exact query terms. This is particularly useful for
    cross-disciplinary landscape searches where terminology varies.

    Only papers that have a PubMed ID are included in the results because
    get_pmcid() and fetch_abstract() — used downstream — require a PMID.

    Args:
        query (str): Search query — typically the anchor's core_question or
                     key terms extracted from it.
        limit (int): Maximum number of results to return. Default 15.

    Returns:
        list[dict]: Normalised paper dicts with relationship="keyword_match".
                    Empty list if the API call fails or returns no PMID-linked
                    papers.
    """
    data = semantic_request(
        "/paper/search",
        {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,externalIds,journal,citationCount",
        },
    )

    if not data or "data" not in data:
        return []

    results = []
    for paper in data["data"]:
        # Skip papers with no PubMed ID — they can't enter the pipeline.
        external_ids = paper.get("externalIds") or {}
        pmid = external_ids.get("PubMed")
        if not pmid:
            continue

        # Format as "Surname et al." for multi-author papers, or full name
        # for single-author papers.
        authors_list = paper.get("authors") or []
        if authors_list:
            first_author = authors_list[0].get("name", "")
            # Split on space and take the last token as the surname — handles
            # "FirstName LastName" and "F. LastName" formats.
            surname = first_author.split(" ")[-1]
            authors_str = (
                f"{surname} et al." if len(authors_list) > 1 else first_author
            )
        else:
            authors_str = "Unknown"

        # journal may be a dict {"name": "..."} or None — guard both cases.
        journal_data = paper.get("journal") or {}
        journal = journal_data.get("name", "") if isinstance(journal_data, dict) else ""

        results.append(normalise_paper(
            pmid=pmid,
            title=paper.get("title", ""),
            authors=authors_str,
            journal=journal,
            year=paper.get("year") or 0,
            relationship="keyword_match",
        ))

    return results


def semantic_citations(pmid: str, limit: int = 20) -> list[dict]:
    """
    Fetches papers that cite the given paper via Semantic Scholar.

    Forward citation traversal — returns papers published after the anchor that
    build on it. More comprehensive than PubMed elink because Semantic Scholar
    indexes preprints and conference papers in addition to journal articles.

    The citation intent field (background, methodology, result) is preserved
    in the normalised dict so Agent 3 can distinguish papers that adopt the
    anchor's methods from those that merely cite it as background.

    Args:
        pmid (str): PubMed ID of the anchor paper.
        limit (int): Maximum number of citing papers to return. Default 20.

    Returns:
        list[dict]: Normalised paper dicts with relationship="cited_by_anchor".
                    Papers without a PubMed ID are silently skipped.
    """
    data = semantic_request(
        f"/paper/PMID:{pmid}/citations",
        {
            "limit": limit,
            # Request fields for the citing paper and the citation intent.
            "fields": (
                "citingPaper.title,citingPaper.authors,"
                "citingPaper.year,citingPaper.externalIds,"
                "citingPaper.journal,intents"
            ),
        },
    )

    if not data or "data" not in data:
        return []

    results = []
    for item in data["data"]:
        paper = item.get("citingPaper") or {}
        external_ids = paper.get("externalIds") or {}
        pmid_result = external_ids.get("PubMed")
        if not pmid_result:
            continue

        # Take the first listed intent as the primary one — papers occasionally
        # have multiple intents (e.g. ["background", "methodology"]).
        intents = item.get("intents") or []
        intent = intents[0] if intents else ""

        authors_list = paper.get("authors") or []
        if authors_list:
            surname = authors_list[0].get("name", "").split(" ")[-1]
            authors_str = (
                f"{surname} et al."
                if len(authors_list) > 1
                else authors_list[0].get("name", "")
            )
        else:
            authors_str = "Unknown"

        journal_data = paper.get("journal") or {}
        journal = journal_data.get("name", "") if isinstance(journal_data, dict) else ""

        results.append(normalise_paper(
            pmid=pmid_result,
            title=paper.get("title", ""),
            authors=authors_str,
            journal=journal,
            year=paper.get("year") or 0,
            relationship="cited_by_anchor",
            intent=intent,
        ))

    return results


def semantic_references(pmid: str, limit: int = 20) -> list[dict]:
    """
    Fetches papers referenced by the given paper via Semantic Scholar.

    Backward citation traversal — returns the foundational literature that the
    anchor paper builds on. These papers populate the "Built upon" section of
    the landscape report and help the user understand the intellectual lineage
    of the anchor's methods and claims.

    Args:
        pmid (str): PubMed ID of the anchor paper.
        limit (int): Maximum number of references to return. Default 20.

    Returns:
        list[dict]: Normalised paper dicts with relationship="cites_anchor".
                    Papers without a PubMed ID are silently skipped.
    """
    data = semantic_request(
        f"/paper/PMID:{pmid}/references",
        {
            "limit": limit,
            "fields": (
                "citedPaper.title,citedPaper.authors,"
                "citedPaper.year,citedPaper.externalIds,"
                "citedPaper.journal,intents"
            ),
        },
    )

    if not data or "data" not in data:
        return []

    results = []
    for item in data["data"]:
        paper = item.get("citedPaper") or {}
        external_ids = paper.get("externalIds") or {}
        pmid_result = external_ids.get("PubMed")
        if not pmid_result:
            continue

        intents = item.get("intents") or []
        intent = intents[0] if intents else ""

        authors_list = paper.get("authors") or []
        if authors_list:
            surname = authors_list[0].get("name", "").split(" ")[-1]
            authors_str = (
                f"{surname} et al."
                if len(authors_list) > 1
                else authors_list[0].get("name", "")
            )
        else:
            authors_str = "Unknown"

        journal_data = paper.get("journal") or {}
        journal = journal_data.get("name", "") if isinstance(journal_data, dict) else ""

        results.append(normalise_paper(
            pmid=pmid_result,
            title=paper.get("title", ""),
            authors=authors_str,
            journal=journal,
            year=paper.get("year") or 0,
            relationship="cites_anchor",
            intent=intent,
        ))

    return results


def pubmed_author_search(author_name: str, limit: int = 10) -> list[dict]:
    """
    Searches PubMed for recent papers by a given author.

    Used to build the author network component of the literature scout —
    finding other papers in the same research thread as the anchor paper's
    authors. Results are restricted to the last 10 years to keep the landscape
    current and avoid surfacing early-career work that predates the author's
    current research focus.

    Author name format: "Surname Initial" (e.g. "Teles J") is the most
    reliable format for PubMed author search. Full names also work but
    may return fewer results due to inconsistent indexing.

    Args:
        author_name (str): Author name in PubMed format, e.g. "Teles J".
        limit (int):       Maximum number of papers to return. Default 10.

    Returns:
        list[dict]: Normalised paper dicts with relationship="same_author".
                    Title is extracted from the first non-empty line of the
                    abstract text (a heuristic that works for most PubMed
                    records). Returns an empty list if the search fails or
                    the author has no indexed papers in the time window.
    """
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": f"{author_name}[Author]",
        "datetype": "pdat",
        "reldate": 3650,   # 10 years in days
        "retmax": limit,
        "retmode": "json",
    }

    try:
        response = requests.get(search_url, params=search_params, timeout=10)
        pmids = (
            response.json()
            .get("esearchresult", {})
            .get("idlist", [])
        )
    except Exception:
        return []

    if not pmids:
        return []

    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    results = []

    for pmid in pmids:
        try:
            fetch_response = requests.get(
                fetch_url,
                params={
                    "db": "pubmed",
                    "id": pmid,
                    "rettype": "abstract",
                    "retmode": "text",
                },
                timeout=10,
            )
            abstract_text = fetch_response.text.strip()

            # PubMed plain-text abstract format (rettype=abstract, retmode=text)
            # returns records in roughly this order:
            #
            #   Journal. Year Month Day;vol(issue):pages. doi: 10.xxxx/...
            #
            #   Title of the paper — this is what we want.
            #
            #   Authors AN, Author2 BN.
            #
            #   [Abstract body...]
            #
            # The first non-empty line is usually the citation header, not the
            # title. We scan all non-empty lines and skip anything that looks
            # like a citation, date, or DOI line using four filters:
            #
            #   len > 30:               skip short lines (journal abbrevs, initials)
            #   not line[0].isdigit():  skip numbered references (e.g. "1. J Biol...")
            #   "doi" not in line:      skip lines containing a DOI
            #   not \d{4};\d:           skip volume/issue format "2026;8(1):..."
            #   not \d{4} \w+ \d+:      skip date format "2026 Apr 8"
            lines = [l.strip() for l in abstract_text.split("\n") if l.strip()]
            title = f"PMID {pmid}"
            for line in lines:
                if (
                    len(line) > 30
                    and not line[0].isdigit()
                    and "doi" not in line.lower()
                    and not re.search(r'\d{4};\d', line)
                    and not re.search(r'\d{4} \w+ \d+', line)
                ):
                    title = line
                    break

            results.append(normalise_paper(
                pmid=pmid,
                title=title,
                authors=author_name,
                journal="",
                year=0,
                relationship="same_author",
            ))
        except Exception:
            # Skip individual papers that fail to fetch rather than aborting
            # the entire author search.
            continue

    return results


def check_full_text_batch(
    papers: list[dict],
    batch_size: int = 20,
) -> list[dict]:
    """
    Refreshes PMC full-text availability for a list of papers.

    Called on the top-N ranked papers before the curation screen so the UI
    can show accurate "Full text available" / "Abstract only" badges. Running
    this on all candidates during scouting would add too much latency; calling
    it on the ranked shortlist keeps the wait acceptable.

    Papers are processed in batches of batch_size, each batch fully parallel
    via ThreadPoolExecutor. With the default batch_size of 20 and max_workers
    capped at 20, up to 20 PMC lookups run concurrently per batch.

    Args:
        papers (list[dict]): Normalised paper dicts, each with a "pmid" key.
        batch_size (int):    Papers per parallel batch. Default 20.

    Returns:
        list[dict]: The same dicts with has_full_text (bool), pmcid
                    (str or None), and full_text_status updated in place.
    """
    def check_one(paper):
        try:
            pmcid = get_pmcid(paper["pmid"])
            paper["has_full_text"] = pmcid is not None
            paper["pmcid"] = pmcid
            paper["full_text_status"] = (
                "available" if pmcid is not None
                else "unavailable"
            )
        except Exception:
            paper["has_full_text"] = False
            paper["pmcid"] = None
            paper["full_text_status"] = "unknown"
        return paper

    results = []
    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        with ThreadPoolExecutor(
            max_workers=min(len(batch), 20)
        ) as ex:
            batch_results = list(ex.map(check_one, batch))
        results.extend(batch_results)
    return results


def literature_scout(anchor: dict) -> list:
    """
    Agent 1: Fetches candidate papers from multiple sources based on the anchor.

    Activates up to three retrieval strategies depending on the anchor's
    search_strategy flags:
      - keyword_search:     semantic_search() on the anchor's core_question
      - citation_traversal: semantic_citations() + semantic_references() on the
                            seed PMID — forward and backward traversal
      - author_network:     pubmed_author_search() for the first author of the
                            anchor paper (extracted from its PubMed abstract)

    Rate-limit strategy:
        Semantic Scholar tools (semantic_search, semantic_citations,
        semantic_references) share a single rate limit of ~1 req/sec.
        Running them concurrently would cause all three to 429 simultaneously,
        and the retries in semantic_request() would keep colliding on the same
        backoff cadence. They are therefore run sequentially with a 1 s delay.

        PubMed tools (pubmed_author_search) hit a different API with a more
        generous rate limit and can safely run in parallel.

    Args:
        anchor (dict): Anchor document from anchor_builder(), containing
                       core_question, field_scope, search_strategy (dict of
                       boolean flags), and optionally detected_pmid.

    Returns:
        list[dict]: Deduplicated candidate papers (by PMID), each a normalised
                    paper dict with keys: pmid, title, authors, journal, year,
                    relationship, intent, has_full_text, pmcid.
                    Returns [] if no tasks are enabled or all tasks fail.
    """
    strategy = anchor.get("search_strategy", {})
    detected_pmid = anchor.get("detected_pmid")
    core_question = anchor.get("core_question", "")

    # ── Build task list ───────────────────────────────────────────────────────
    # Each task is a (func, args_tuple, kwargs_dict) triple so the executor and
    # sequential loop can call them uniformly with func(*args, **kwargs).
    tasks = []

    if strategy.get("keyword_search", True) and core_question:
        tasks.append((semantic_search, (core_question,), {}))

    if strategy.get("citation_traversal", False) and detected_pmid:
        tasks.append((semantic_citations, (detected_pmid,), {}))
        tasks.append((semantic_references, (detected_pmid,), {}))

    if strategy.get("author_network", True) and detected_pmid:
        # Retrieve the first author of the anchor paper from its PubMed abstract
        # so pubmed_author_search has a concrete name to query.
        # The abstract text format puts the author list early; we scan the first
        # five non-empty lines for a "Surname Initials" pattern.
        try:
            abstract_text = fetch_abstract(detected_pmid)
            lines = [l.strip() for l in abstract_text.split("\n") if l.strip()]
            for line in lines[:5]:
                author_match = re.match(r'^([A-Z][a-z]+\s+[A-Z]{1,3})\b', line)
                if author_match:
                    tasks.append(
                        (pubmed_author_search, (author_match.group(1),), {})
                    )
                    break
        except Exception:
            pass  # Author network is best-effort; skip if extraction fails

    if not tasks:
        return []

    all_results = []

    # Split tasks into semantic scholar and pubmed groups
    semantic_tasks = []
    pubmed_tasks = []
    for task in tasks:
        func, args, kwargs = task
        if func.__name__ in [
            'semantic_search',
            'semantic_citations',
            'semantic_references'
        ]:
            semantic_tasks.append(task)
        else:
            pubmed_tasks.append(task)

    # Run semantic tasks sequentially with 1s delay
    # Semantic Scholar rate limit is 1 req/sec
    for i, task in enumerate(semantic_tasks):
        func, args, kwargs = task
        try:
            results = func(*args, **kwargs)
            all_results.extend(results)
            print(f"[scout] {func.__name__}: {len(results)} results")
        except Exception as e:
            print(f"[scout] {func.__name__} failed: {e}")
        if i < len(semantic_tasks) - 1:
            time.sleep(1)

    # Run pubmed tasks in parallel
    def run_task(task):
        func, args, kwargs = task
        return func(*args, **kwargs)

    if pubmed_tasks:
        with ThreadPoolExecutor(
            max_workers=len(pubmed_tasks)
        ) as ex:
            futures = {
                ex.submit(run_task, task): task
                for task in pubmed_tasks
            }
            for future in as_completed(futures):
                result = future.result()
                all_results.extend(result)

    # ── Deduplicate by PMID ───────────────────────────────────────────────────
    # Multiple tools can return the same paper (e.g. a highly-cited paper
    # appears in both keyword search and citation traversal). Keep the first
    # occurrence to preserve the relationship label from the tool that found it.
    seen_pmids: set = set()
    unique_results = []
    for paper in all_results:
        if paper["pmid"] not in seen_pmids:
            seen_pmids.add(paper["pmid"])
            unique_results.append(paper)

    return unique_results


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

    """
    if not candidates:
        return candidates

    candidate_lines = "\n".join(
        f'  {{"pmid": "{p["pmid"]}", "title": {json.dumps(p.get("title", ""))}, '
        f'"authors": {json.dumps(p.get("authors", ""))}, "year": {p.get("year", 0)}, '
        f'"relationship": "{p.get("relationship", "")}", "intent": "{p.get("intent", "")}"}},'
        for p in candidates
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=(
            "You are a systematic literature reviewer. "
            "Score each paper 0-100 for relevance to the research anchor. "
            "Return only a JSON array. No markdown, no preamble."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Research anchor:\n"
                f"  core_question: {json.dumps(anchor.get('core_question', ''))}\n"
                f"  field_scope: {json.dumps(anchor.get('field_scope', ''))}\n"
                f"  key_debates: {json.dumps(anchor.get('key_debates', []))}\n\n"
                f"Papers to score:\n[\n{candidate_lines}\n]\n\n"
                "Return a JSON array where each element has:\n"
                '  "pmid": "<same pmid>",\n'
                '  "relevance_score": <integer 0-100>,\n'
                '  "relevance_rationale": "<one sentence>"\n'
                "Score 90-100: directly answers core question; canonical reference.\n"
                "Score 70-89: highly relevant; covers key methods or debates in scope.\n"
                "Score 50-69: peripherally relevant; useful context but not central.\n"
                "Score <50: off-topic or outside field_scope time range."
            ),
        }],
    )

    raw = next((b.text for b in response.content if b.type == "text"), "[]")
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    try:
        scores = json.loads(raw)
        if not isinstance(scores, list):
            scores = []
    except json.JSONDecodeError:
        scores = []

    # Build score lookup by pmid
    score_map = {}
    for s in scores:
        if isinstance(s, dict) and "pmid" in s:
            score_map[str(s["pmid"])] = s

    # Apply scores to candidates
    # Use .get() with defaults so missing scores never cause KeyError
    for paper in candidates:
        score_data = score_map.get(str(paper["pmid"]), {})
        paper["relevance_score"] = score_data.get(
            "relevance_score", 50
        )
        paper["relevance_rationale"] = score_data.get(
            "relevance_rationale", "Score unavailable"
        )

    # Ensure every paper has relevance_score before sorting
    # This prevents KeyError if any paper was missed
    for paper in candidates:
        if "relevance_score" not in paper:
            paper["relevance_score"] = 50
            paper["relevance_rationale"] = "Score unavailable"

    return sorted(candidates, key=lambda p: p["relevance_score"], reverse=True)


def synthesis_agent(
    anchor: dict,
    selected_papers: list[dict],
    paper_pdfs: dict,
) -> str:
    """
    Agent 3: Generates a structured literature landscape report from
    selected papers and the anchor document.

    For each selected paper, fetches the richest available content in
    priority order:
      1. Uploaded PDF (richest — includes figures and full structure)
      2. PMC full text (good — complete article text)
      3. Abstract (fallback — limited but always available)

    All paper content is assembled into a single Claude API call with
    the anchor as the organising frame, producing a six-section markdown
    report structured around the anchor's core question and key debates.

    Args:
        anchor (dict):              Confirmed anchor document with
                                    core_question, field_scope,
                                    key_debates, detected_pmid.
        selected_papers (list[dict]): Curated paper list, each with
                                    pmid, title, relationship,
                                    has_full_text, full_text_status.
        paper_pdfs (dict):          pmid → pdf bytes for any papers
                                    where the user uploaded a PDF.

    Returns:
        str: Structured markdown landscape report ready for display
             in the UI or export as a document.
    """
    from research_agent import fetch_full_text, fetch_abstract

    # ── Step 1: Fetch the richest available content for each paper ────────
    context_parts = []

    for paper in selected_papers:
        pmid = paper["pmid"]
        title = paper["title"]
        relationship = paper["relationship"]

        # Priority 1: user-uploaded PDF — summarise_pdf extracts structured
        # content including figures that plain-text routes would miss.
        if pmid in paper_pdfs and paper_pdfs[pmid]:
            try:
                from research_agent import summarise_pdf
                summary, _, _, _ = summarise_pdf(paper_pdfs[pmid])
                content = summary
                source = "uploaded PDF"
            except Exception:
                content = fetch_abstract(pmid)
                source = "abstract (PDF processing failed)"

        # Priority 2: PMC full text — complete article via eutils efetch.
        elif paper.get("has_full_text"):
            try:
                content = fetch_full_text(pmid)
                source = "full text"
            except Exception:
                content = fetch_abstract(pmid)
                source = "abstract (full text fetch failed)"

        # Priority 3: abstract fallback — always available from PubMed.
        else:
            content = fetch_abstract(pmid)
            source = "abstract"

        # Truncate to 2000 chars to manage context window — long enough for
        # key content, short enough to fit many papers in a single API call.
        if len(content) > 2000:
            content = content[:2000] + "...[truncated]"

        context_parts.append(
            f"PAPER: {title}\n"
            f"PMID: {pmid}\n"
            f"RELATIONSHIP TO ANCHOR: {relationship}\n"
            f"SOURCE: {source}\n"
            f"CONTENT:\n{content}"
        )

    full_context = "\n\n---\n\n".join(context_parts)

    # Format key debates as a bullet list for the prompt.
    debates_str = "\n".join(
        f"- {d}" for d in anchor.get("key_debates", [])
    )

    # ── Step 2: Single Claude call to synthesise the landscape ────────────
    # The anchor is the organising frame, not just context — every section
    # of the report is written through the lens of the core_question.
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        system=(
            "You are a research synthesis expert helping a scientist "
            "understand the literature landscape around a specific "
            "research question.\n\n"
            "Write in clear, precise scientific prose. Be specific — "
            "name papers, authors, and findings. Avoid vague generalities. "
            "Structure your response with markdown headers exactly as requested."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Synthesise a literature landscape from the papers below.\n\n"
                f"ANCHOR QUESTION: {anchor.get('core_question', '')}\n"
                f"FIELD SCOPE: {anchor.get('field_scope', '')}\n"
                f"KEY DEBATES:\n{debates_str}\n\n"
                f"PAPERS ({len(selected_papers)} total):\n\n"
                f"{full_context}\n\n"
                "Generate a structured landscape report with EXACTLY these "
                "six sections in this order. Use these exact markdown headers:\n\n"
                "## Context: what this field builds on\n"
                "Summarise the foundational work these papers collectively build "
                "on. What were the key prior findings and methods? Be specific — "
                "name the foundational contributions.\n\n"
                "## Central contribution\n"
                "What is the central advance represented by the anchor paper and "
                "closely related work? What gap did it fill? What was genuinely new?\n\n"
                "## What has been built on it\n"
                "Which papers cite or extend the anchor work? What did they add, "
                "modify, or challenge? How has the field moved forward?\n\n"
                "## Parallel approaches\n"
                "What other approaches address the same core question? How do they "
                "differ methodologically or conceptually? What are the tradeoffs "
                "between approaches?\n\n"
                "## Open questions\n"
                "What remains unresolved? What are the key limitations of current "
                "work? What would the most important next experiment or analysis be?\n\n"
                "## Recommended reading order\n"
                "List 4-6 papers from those provided in the order a new reader "
                "should encounter them to build understanding progressively. For "
                "each paper give one sentence on why it comes at that point in "
                "the sequence.\n\n"
                "Write in flowing prose within each section. Be specific and cite "
                "paper titles and authors. Do not add any sections beyond these six."
            ),
        }],
    )

    return next(
        (b.text for b in response.content if b.type == "text"),
        "Unable to generate landscape report.",
    )
