"""
research_agent.py — Core agent logic for the Research Assistant application.

This module handles all interaction with the PubMed/PMC APIs and drives the
Claude tool-use agent loop that fetches and summarises scientific papers.

Relationship to app.py:
    app.py imports run_agent() from this module. app.py owns the Streamlit UI
    and the follow-up chat loop; this module owns paper retrieval and the
    initial summarisation agent.

Main flow of execution:
    1. run_agent() sends a user request to Claude with the fetch_full_text tool
       available.
    2. Claude decides to call fetch_full_text, returning a tool_use block.
    3. We execute fetch_full_text() locally, which calls get_pmcid() then the
       PMC efetch endpoint (or falls back to the abstract endpoint).
    4. We send the tool result back to Claude in a second API call.
    5. Claude produces the final summary text, which run_agent() returns.
"""

import anthropic
import os
import json
import sys
import base64
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY (and any other vars) from a local .env file so the
# module works both in development (no env vars set) and in production.
load_dotenv()

# Initialise the Anthropic client once at module load time rather than inside
# every function to avoid the overhead of creating a new HTTP session per call.
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def fetch_abstract(pubmed_id: str) -> str:
    """
    Fetches the abstract text for a paper from the PubMed efetch API.

    This is the fallback used when full PMC text is unavailable. It hits the
    PubMed efetch endpoint and asks for the abstract in plain-text format.

    Args:
        pubmed_id (str): The PubMed ID of the paper, e.g. "23990771".

    Returns:
        str: The full abstract as a plain-text string, exactly as returned by
             the PubMed API (may include title, authors, and journal metadata).
             Raises requests.HTTPError if the request fails (non-2xx status).

    Example return:
        "Alzheimer Dis Assoc Disord. 2013 Jul-Sep;27(3):260-6. doi: 10.1097/..."
    """
    # Build the request parameters for the PubMed efetch endpoint.
    # rettype=abstract returns structured abstract text.
    # retmode=text asks for plain text rather than XML or JSON.
    response = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={"db": "pubmed", "id": pubmed_id, "rettype": "abstract", "retmode": "text"},
    )
    # Raise immediately if PubMed returns a 4xx or 5xx status code so the
    # caller gets a clear HTTPError rather than silently returning empty text.
    response.raise_for_status()
    return response.text


def get_pmcid(pmid: str) -> str | None:
    """
    Checks whether a paper has full text in PubMed Central.

    Uses PMC esearch API which is significantly faster than
    the elink API. Searches PMC database directly for the
    given PMID and returns the PMCID if found.

    Args:
        pmid (str): PubMed ID to check

    Returns:
        str: PMCID e.g. "PMC3749951" if found
        None: if not in PMC or request fails
    """
    url = (
        "https://eutils.ncbi.nlm.nih.gov"
        "/entrez/eutils/esearch.fcgi"
    )
    params = {
        "db": "pmc",
        "term": f"{pmid}[PMID]",
        "retmode": "json",
        "retmax": 1,
    }

    try:
        response = requests.get(url, params=params, timeout=8)
        data = response.json()
        ids = (
            data
            .get("esearchresult", {})
            .get("idlist", [])
        )
        if ids:
            # PMC IDs from esearch don't have PMC prefix —
            # add it for consistency with rest of pipeline
            return f"PMC{ids[0]}"
        return None
    except Exception:
        return None


def fetch_full_text(pmid: str) -> str:
    """
    Fetches the best available text for a paper: full PMC text if accessible,
    otherwise the PubMed abstract.

    The function first resolves the PMID to a PMCID via get_pmcid(). If a PMCID
    exists it downloads the full-text XML from PMC and parses out the abstract,
    introduction, methods, results, discussion, and figure captions into labelled
    sections. If any step fails it falls back to fetch_abstract().

    Args:
        pmid (str): The PubMed ID of the paper, e.g. "37651234".

    Returns:
        str: A multi-section plain-text string with labelled headers
             (ABSTRACT, INTRODUCTION, METHODS, RESULTS, DISCUSSION, FIGURES)
             followed by a provenance footer. Falls back to the abstract text
             with "[Source: Abstract only…]" appended when full text is
             unavailable.

    Example return:
        "ABSTRACT\nBackground: ...\n\nINTRODUCTION\n...\n\n[Source: Full text via PubMed Central]"
    """
    # Attempt to find a PMC deposit for this PMID. If none exists, skip
    # straight to the abstract fallback at the bottom of the function.
    pmcid = get_pmcid(pmid)
    if pmcid:
        try:
            # Download the full article XML from PubMed Central.
            # rettype=full + retmode=xml returns the complete JATS/NLM XML.
            response = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pmc", "id": pmcid, "rettype": "full", "retmode": "xml"},
            )
            response.raise_for_status()

            # Parse the XML string into an ElementTree for XPath queries.
            root = ET.fromstring(response.text)

            # Verify the XML belongs to the requested paper.
            # PMC's elink mapping can occasionally return the wrong PMCID.
            # Cross-check by reading the PMID embedded in the article XML itself.
            pmid_tag = root.find('.//article-id[@pub-id-type="pmid"]')
            if pmid_tag is not None:
                if pmid_tag.text and pmid_tag.text.strip() != pmid:
                    print(
                        f"[warning] XML PMID mismatch: requested {pmid}, "
                        f"got {pmid_tag.text.strip()}. Falling back to abstract."
                    )
                    return fetch_abstract(pmid) + "\n\n[Source: Abstract only - PMID mismatch]"
            else:
                # No PMID tag means we cannot confirm the record matches —
                # treat as unverified and fall back rather than risk wrong content.
                print("[warning] No PMID tag found in XML — falling back to abstract")
                return fetch_abstract(pmid) + "\n[Source: Abstract only — PMC record could not be verified]"

            # Accumulate named sections; we join them later with double newlines.
            sections = []

            # ── Extract the abstract ─────────────────────────────────────────
            # JATS XML may contain multiple <abstract> elements (structured or
            # plain). Concatenate all of their text nodes into one block.
            abstracts = root.findall(".//abstract")
            if abstracts:
                text = " ".join(" ".join(node.itertext()) for node in abstracts).strip()
                if text:
                    sections.append(f"ABSTRACT\n{text}")

            # ── Extract named body sections ──────────────────────────────────
            # Each tuple maps a display label to a lambda that recognises
            # section titles used by that section in real-world PMC articles.
            for label, keywords in [
                ("INTRODUCTION", lambda t: t in ("Introduction", "Background")),
                ("METHODS", lambda t: "Method" in t or "Material" in t),
                ("RESULTS", lambda t: "Result" in t),
                ("DISCUSSION", lambda t: "Discussion" in t),
            ]:
                # Scan every <sec> element in the document looking for a
                # <title> whose text satisfies the keyword predicate. We stop
                # at the first match (break) so we don't duplicate content
                # when a paper has e.g. both "Results" and "Results and Discussion".
                for sec in root.findall(".//sec"):
                    title_el = sec.find("title")
                    if title_el is not None and title_el.text and keywords(title_el.text.strip()):
                        # itertext() walks the whole subtree, so we capture
                        # nested paragraphs, list items, etc. without needing
                        # explicit recursion.
                        text = " ".join(sec.itertext()).strip()
                        if text:
                            sections.append(f"{label}\n{text}")
                        break

            # ── Extract figure captions ──────────────────────────────────────
            # Include figure labels and captions so Claude can reference them
            # when summarising experimental results.
            figs = root.findall(".//fig")
            if figs:
                fig_lines = []
                for fig in figs:
                    label_el = fig.find("label")   # e.g. "Figure 1"
                    caption_el = fig.find("caption")  # full caption paragraph(s)
                    label_text = " ".join(label_el.itertext()).strip() if label_el is not None else ""
                    caption_text = " ".join(caption_el.itertext()).strip() if caption_el is not None else ""
                    # Join label and caption with ": ", skipping whichever is empty.
                    entry = ": ".join(x for x in [label_text, caption_text] if x)
                    if entry:
                        fig_lines.append(entry)
                if fig_lines:
                    sections.append("FIGURES\n" + "\n".join(fig_lines))

            # Tiered fallback for non-standard section structure.
            # Try <body> text directly before resorting to the full-tree catch-all.
            if not sections:
                body_el = root.find(".//body")
                if body_el is not None:
                    body_chunks = [
                        chunk.strip()
                        for el in body_el.iter()
                        for chunk in (el.text, el.tail)
                        if chunk and len(chunk.strip()) > 30
                    ]
                    if body_chunks:
                        sections.append(" ".join(body_chunks))

            # Last-resort catch-all: walk every element in the entire tree.
            # Less structured than body extraction but guarantees we return
            # something when the XML has content but non-standard tags.
            if not sections:
                raw_chunks = [
                    chunk.strip()
                    for el in root.iter()
                    for chunk in (el.text, el.tail)
                    if chunk and len(chunk.strip()) > 20
                ]
                if raw_chunks:
                    sections.append(" ".join(raw_chunks))

            # Join sections with blank lines and append a provenance footer
            # so Claude and the user know the text came from PMC full text.
            output = "\n\n".join(s for s in sections if s)

            # Empty or near-empty output means the XML existed but contained no
            # usable prose (e.g. a metadata-only record). Fall back to the
            # abstract so Claude always has something meaningful to summarise.
            if len(output) < 200:
                return fetch_abstract(pmid) + "\n\n[Source: Abstract only - full text parse failed]"

            return output + "\n\n[Source: Full text via PubMed Central]"
        except Exception:
            # Any XML parse error, HTTP error, or unexpected structure means
            # we fall back gracefully rather than crashing the agent loop.
            return fetch_abstract(pmid) + "\n\n[Source: Abstract only - full text not available]"
    else:
        # No PMCID found — the paper has no free full-text deposit in PMC.
        return fetch_abstract(pmid) + "\n\n[Source: Abstract only - full text not available]"


# ── Tool definitions ─────────────────────────────────────────────────────────
#
# Tool definitions are JSON Schema objects that describe callable functions to
# Claude. When Claude receives a request alongside a tools list it decides
# whether to answer directly or to call one of the tools. If it calls a tool,
# the API returns stop_reason="tool_use" and a tool_use content block
# containing the tool name and the arguments Claude chose to pass.
#
# The structure of each definition follows the Anthropic tool-use spec:
#   name        — the identifier Claude uses to refer to the tool
#   description — natural-language explanation that helps Claude decide WHEN
#                 to use the tool; more detail → better decisions
#   input_schema — JSON Schema for the arguments; Claude uses this to know
#                  what inputs to supply when calling the tool
#
# We define only one tool here because the single fetch_full_text function
# covers both full-text and abstract retrieval, making the agent's decision
# simple: always call this tool before summarising.
tools = [
    {
        "name": "fetch_full_text",
        "description": (
            "Fetches the full text of a paper from PubMed Central if available, "
            "otherwise fetches the abstract. "
            "Returns the best available text for the paper."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pubmed_id": {
                    "type": "string",
                    "description": "The PubMed ID of the paper",
                }
            },
            "required": ["pubmed_id"],
        },
    }
]


def run_agent(pubmed_id: str) -> tuple[str, str]:
    """
    Runs the two-turn Claude tool-use agent loop and returns a summary plus
    the raw fetched text.

    Turn 1 — Intent: sends the user's request to Claude with the
    fetch_full_text tool available. Claude is expected to respond with
    stop_reason="tool_use", indicating it wants to call fetch_full_text before
    answering.

    Turn 2 — Ground truth: executes fetch_full_text locally, then sends the
    tool result back to Claude. Claude now has the paper text in context and
    produces the final summary with stop_reason="end_turn".

    If for any reason Claude answers directly in turn 1 (stop_reason="end_turn"
    with no tool call), we return that text with an empty fetched_text string.

    Args:
        pubmed_id (str): The PubMed ID of the paper to summarise, e.g. "37651234".

    Returns:
        tuple[str, str]: A two-element tuple:
            [0] summary (str)    — Claude's markdown summary of the paper.
            [1] fetched_text (str) — The raw text returned by fetch_full_text,
                                     used by app.py to show a provenance caption.
                                     Empty string if Claude did not call the tool.

    Example return:
        ("**Title:** ...\n**Authors:** ...\n...", "ABSTRACT\n...\n\n[Source: Full text via PubMed Central]")
    """
    # ── Turn 1: ask Claude to fetch and summarise the paper ──────────────────
    #
    # At this point messages contains a single user turn. We pass the tools
    # list so Claude knows fetch_full_text is available.
    #
    # Expected stop_reason: "tool_use" — Claude decides it needs the paper text
    # before it can write a summary and signals that by returning a tool_use
    # content block.
    #
    # Unexpected stop_reason: "end_turn" — Claude answered without calling the
    # tool (e.g. it already knows the paper). We handle this in the fallback
    # at the end of the function.
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": f"Can you fetch and summarise the paper for PubMed ID {pubmed_id}?"}
        ],
    )

    if response.stop_reason == "tool_use":
        # Extract the tool_use content block. There will be exactly one because
        # we only defined one tool and the request implies a single paper lookup.
        # next() raises StopIteration if no match found —
        # always provide a default when the API response
        # structure cannot be guaranteed
        tool_block = next((b for b in response.content if b.type == "tool_use"), None)
        if not tool_block:
            return "Unable to generate summary — the model returned an unexpected response. Please try again.", ""

        # Claude may normalise or re-interpret the PMID (e.g. strip whitespace),
        # so we use the ID that Claude passed to the tool rather than the raw
        # pubmed_id argument.
        resolved_id = tool_block.input["pubmed_id"]

        # Execute the tool call locally — this is the step the API cannot do
        # for us; we run the Python function and collect its return value.
        fetched_text = fetch_full_text(resolved_id)

        # ── Turn 2: send the tool result back and get the final summary ──────
        #
        # The messages list now contains three entries:
        #   1. user:      the original request
        #   2. assistant: response.content (the tool_use block from turn 1)
        #   3. user:      a tool_result block wrapping the fetched text
        #
        # The tool_result block must include the tool_use_id from turn 1 so the
        # API can match it to the correct tool call.
        #
        # Expected stop_reason: "end_turn" — Claude now has the paper text and
        # produces a complete summary with no further tool calls needed.
        final_response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            tools=tools,
            messages=[
                {"role": "user", "content": f"Can you fetch and summarise the paper for PubMed ID {pubmed_id}?"},
                # Pass the full content list from turn 1 (not just the text) so
                # the assistant turn correctly includes the tool_use block.
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            # Link this result to the specific tool call that
                            # requested it so the API can correlate them.
                            "tool_use_id": tool_block.id,
                            "content": fetched_text,
                        }
                    ],
                },
            ],
        )

        # If the model refused the full-text content, retry with just the abstract
        # so the user still receives a useful summary.
        if final_response.stop_reason == "refusal":
            abstract = fetch_abstract(pubmed_id)
            refusal_response = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": f"Please summarise this paper abstract:\n\n{abstract}"}
                ],
            )
            # next() raises StopIteration if no match found —
            # always provide a default when the API response
            # structure cannot be guaranteed
            summary = next((b.text for b in refusal_response.content if b.type == "text"), None)
            if not summary:
                summary = "Unable to generate summary — the model returned an unexpected response. Please try again."
            return summary, abstract

        # Extract the plain-text summary from the final response. We skip any
        # non-text blocks (e.g. residual tool_use blocks) with the type check.
        # next() raises StopIteration if no match found —
        # always provide a default when the API response
        # structure cannot be guaranteed
        summary = next((b.text for b in final_response.content if b.type == "text"), None)
        if not summary:
            summary = "Unable to generate summary — the model returned an unexpected response. Please try again."
        return summary, fetched_text

    # Fallback: Claude answered without calling the tool (stop_reason="end_turn").
    # Return whatever text Claude produced and an empty fetched_text so the
    # caller's provenance caption is simply omitted.
    summary = next((b.text for b in response.content if b.type == "text"), "")
    return summary, ""


def fetch_summary(pdf_block: dict, client) -> str:
    """
    Fetches a structured narrative summary from Claude.

    Designed to run in parallel with fetch_info_card() via ThreadPoolExecutor,
    cutting the combined latency of both calls roughly in half.

    Args:
        pdf_block (dict): Base64-encoded PDF document block for the Claude API.
        client: Anthropic client instance — passed explicitly rather than using
                the module-level singleton so this is safe to call from threads.

    Returns:
        str: Markdown-formatted summary with fixed section headers.
    """
    # claude-sonnet-4-6: strong instruction-following for long-form structured
    # output; faster and cheaper than Opus while still producing detailed summaries.
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=(
            "You are a research assistant helping a computational biologist "
            "understand scientific papers. Be precise and structured."
        ),
        messages=[{
            "role": "user",
            "content": [
                pdf_block,
                {
                    "type": "text",
                    "text": (
                        "Provide a structured summary with exactly these sections:\n\n"
                        "**Title:**\n"
                        "**Authors:**\n"
                        "**Journal & Year:**\n\n"
                        "**Problem addressed:**\n"
                        "What gap or question does this paper tackle?\n\n"
                        "**Approach:**\n"
                        "What methods, datasets, and experimental systems?\n\n"
                        "**Key findings:**\n"
                        "What did they find? Be specific with numbers.\n\n"
                        "**Figures worth noting:**\n"
                        "Which figures best support the main claims?\n\n"
                        "**Limitations:**\n"
                        "Main weaknesses or caveats.\n\n"
                        "**Relevance to drug discovery or computational biology:**\n"
                        "Why does this matter?"
                    ),
                },
            ],
        }],
    )
    return next(
        (b.text for b in response.content if b.type == "text"),
        "Unable to generate summary.",
    )


def fetch_info_card(pdf_block: dict, client) -> dict:
    """
    Extracts a structured JSON info card from the paper via Claude.

    Designed to run in parallel with fetch_summary() via ThreadPoolExecutor.
    The system prompt explicitly forbids markdown wrapping so the response can
    be parsed directly with json.loads() without stripping fences.

    Args:
        pdf_block (dict): Base64-encoded PDF document block for the Claude API.
        client: Anthropic client instance — passed explicitly for thread safety.

    Returns:
        dict: Structured paper metadata with keys: title, authors, journal,
              year, hypothesis, model_system, sample_size, statistical_tests,
              key_findings, limitations, datasets_tools,
              relevance_to_drug_discovery. Returns {} if JSON parsing fails.
    """
    # claude-sonnet-4-6: reliable constrained output for structured extraction
    # tasks; Opus would be overkill for a fixed JSON schema response.
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=(
            "You are a precise data extractor. "
            "Return only valid JSON. No markdown, no backticks, no preamble."
        ),
        messages=[{
            "role": "user",
            "content": [
                pdf_block,
                {
                    "type": "text",
                    "text": (
                        "Extract structured information. Return ONLY this JSON object:\n"
                        "{\n"
                        '    "title": "full paper title",\n'
                        '    "authors": "author list",\n'
                        '    "journal": "journal name",\n'
                        '    "year": "publication year",\n'
                        '    "hypothesis": "one sentence central hypothesis",\n'
                        '    "model_system": "cell lines, organisms, cohorts used",\n'
                        '    "sample_size": "key sample sizes",\n'
                        '    "statistical_tests": ["method1", "method2"],\n'
                        '    "key_findings": ["finding1", "finding2", "finding3"],\n'
                        '    "limitations": ["limitation1", "limitation2"],\n'
                        '    "datasets_tools": ["dataset1", "tool1"],\n'
                        '    "relevance_to_drug_discovery": "one sentence"\n'
                        "}"
                    ),
                },
            ],
        }],
    )

    card_text = next(
        (b.text for b in response.content if b.type == "text"),
        "{}",
    )

    # Strip accidental markdown fences if Claude added them despite the system prompt.
    card_text = card_text.strip()
    if card_text.startswith("```"):
        card_text = card_text.split("\n", 1)[1]
        card_text = card_text.rsplit("```", 1)[0]

    try:
        return json.loads(card_text)
    except json.JSONDecodeError:
        return {}


def interpret_figure(img: dict) -> dict:
    """
    Interprets a single figure using Claude vision.

    Designed to be called in parallel via ThreadPoolExecutor — each invocation
    is independent and uses the module-level client singleton directly (the
    Anthropic client is thread-safe for concurrent read-only API calls).

    Args:
        img (dict): Figure dict with 'bytes' (raw image bytes) and 'ext' keys.

    Returns:
        dict: Same img dict with 'interpretation' key added.
    """
    img_base64 = base64.standard_b64encode(img["bytes"]).decode("utf-8")
    ext = img["ext"]
    # Normalise 'jpg' to 'jpeg' to satisfy the API media type spec.
    media_type = "image/jpeg" if ext == "jpg" else f"image/{ext}"

    try:
        # claude-haiku-4-5-20251001: fastest and most cost-efficient model for
        # short vision tasks; ideal here because many calls run in parallel and
        # each response is capped at 200 tokens — latency matters more than depth.
        fig_response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "This is a figure from a scientific paper. "
                            "In 2-3 sentences: what does it show, what is the "
                            "key pattern or finding, and what biological or "
                            "statistical insight does it provide?"
                        ),
                    },
                ],
            }],
        )
        img["interpretation"] = next(
            (b.text for b in fig_response.content if b.type == "text"),
            "Unable to interpret figure.",
        )
    except Exception:
        img["interpretation"] = "Figure interpretation unavailable."

    return img


def summarise_pdf(pdf_bytes: bytes) -> tuple:
    """
    Summarises a PDF paper using Claude's native PDF reading capability.

    Runs API calls in parallel for efficiency:
      - fetch_summary and fetch_info_card execute simultaneously (2 workers)
      - All figure interpretations execute simultaneously (4 workers)

    Images are extracted via PyMuPDF with two filters applied:
      1. Pages whose first 500 characters contain supplementary keywords are
         skipped entirely, keeping only main-paper figures.
      2. Images smaller than 150×150 px are discarded (logos, icons, decorations).

    Args:
        pdf_bytes (bytes): Raw bytes of the uploaded PDF file.

    Returns:
        tuple: (summary, source_label, figures, info_card)
            summary (str): Markdown-formatted narrative summary.
            source_label (str): Provenance string for display in the UI.
            figures (list[dict]): Sorted largest-first; each dict has keys:
                bytes  (bytes) — raw image bytes
                ext    (str)   — file extension, e.g. 'png'
                width, height  (int) — pixel dimensions
                interpretation (str) — Claude's 2-3 sentence description
            info_card (dict): Structured fields extracted from the paper.
    """
    import fitz  # PyMuPDF — imported here so it's an optional dependency

    # ── Extract images, skipping supplementary pages ──────────────────────────
    # Supplementary keyword detection works by checking the first 500 characters
    # of each page's plain text. This covers the page title and first paragraph,
    # where supplementary headers almost always appear in journal PDFs.
    # Limitation: appendices without explicit header text (e.g. numbered tables
    # in a continuous document) will not be filtered out by this heuristic.
    SUPPLEMENTARY_KEYWORDS = [
        "supplementary",
        "supplemental",
        "supporting information",
        "appendix",
        "extended data",
    ]

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    extracted_images = []
    seen_xrefs: set = set()

    for page in doc:
        # Lowercase for case-insensitive matching — journal PDFs vary in
        # capitalisation of section headers ("Supplementary" vs "SUPPLEMENTARY").
        page_text = page.get_text().lower()

        is_supplementary = any(
            keyword in page_text[:500]
            for keyword in SUPPLEMENTARY_KEYWORDS
        )
        if is_supplementary:
            continue

        for img_ref in page.get_images():
            xref = img_ref[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            img_data = doc.extract_image(xref)
            width = img_data["width"]
            height = img_data["height"]
            # Skip small images — logos, icons, and decorative elements are
            # typically under 150 px in either dimension.
            if width > 150 and height > 150:
                extracted_images.append({
                    "bytes": img_data["image"],
                    "ext": img_data["ext"],
                    "width": width,
                    "height": height,
                })

    # No figure cap — supplementary filtering above controls volume. All main
    # paper figures are forwarded to the parallel interpretation step.

    # ── Encode PDF as base64 for the Claude API ───────────────────────────────
    pdf_base64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    # Reusable document block passed to both summary and info card calls.
    pdf_block = {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": pdf_base64,
        },
    }

    # ── Run summary and info card in parallel ─────────────────────────────────
    # Both calls are fully independent — running them concurrently cuts the
    # combined latency roughly in half compared to sequential execution.
    with ThreadPoolExecutor(max_workers=2) as executor:
        summary_future = executor.submit(fetch_summary, pdf_block, client)
        card_future = executor.submit(fetch_info_card, pdf_block, client)
        summary = summary_future.result()
        info_card = card_future.result()

    # ── Run figure interpretations in parallel ────────────────────────────────
    # Each figure is an independent API call. With 4 workers, total interpretation
    # time scales with ceil(n_figures / 4) instead of n_figures.
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(interpret_figure, img): img
            for img in extracted_images
        }
        results = []
        for future in as_completed(futures):
            results.append(future.result())

    # as_completed() yields in completion order, not submission order.
    # Sort by image area (largest first) as a proxy for figure importance —
    # main paper figures tend to be larger than insets or thumbnails.
    extracted_images = sorted(
        results,
        key=lambda x: x["width"] * x["height"],
        reverse=True,
    )

    return (
        summary,
        "[Source: User uploaded PDF]",
        extracted_images,
        info_card,
    )


def main():
    """
    CLI entry point for running the research agent from the terminal.

    Reads an optional PubMed ID from the first command-line argument and falls
    back to a hard-coded default so the script is runnable without arguments
    during development.

    Args:
        None (reads sys.argv directly).

    Returns:
        None. Prints the summary to stdout.
    """
    # Use the first CLI argument as the PMID, defaulting to "37651234" when
    # none is provided so developers can run `python research_agent.py` without
    # needing to supply an ID every time.
    pubmed_id = sys.argv[1] if len(sys.argv) > 1 else "37651234"
    print("Research agent ready")

    text = fetch_full_text("39486399")
    print(f"fetch_full_text result length: {len(text)} characters")

    # Discard the raw fetched_text (_) — the CLI only needs the final summary.
    summary, _ = run_agent(pubmed_id)
    print(summary)


if __name__ == "__main__":
    main()
