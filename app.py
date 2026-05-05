"""
app.py — Streamlit web UI for the Research Assistant application.

This module owns the browser-facing interface and the follow-up conversation
loop. It imports run_agent() from research_agent.py for the initial paper
fetch and summarisation, then uses a direct Claude API call for every
subsequent chat message.

Relationship to research_agent.py:
    research_agent.py handles all PubMed/PMC API calls and drives the Claude
    tool-use loop that produces the first summary. Once the summary exists,
    this module takes over and manages the ongoing Q&A conversation with the
    user using a simpler stateless messages list.

Main flow of execution:
    1. User enters a PubMed ID and clicks "Fetch & Summarise".
    2. run_agent() is called; the summary and raw fetched text are stored in
       st.session_state so they survive Streamlit reruns.
    3. The summary is rendered in a styled card with provenance info, and two
       download buttons (Markdown and PDF) are shown.
    4. The user can type follow-up questions in the chat input at the bottom.
       Each question is appended to st.session_state.messages, sent to Claude
       with the full conversation history, and the reply is displayed and saved.
"""

import re
import unicodedata
import base64
import json
from datetime import date
import os
import streamlit as st
import anthropic
import fitz  # PyMuPDF — used only to check availability; extraction is in research_agent
from fpdf import FPDF
from research_agent import run_agent, summarise_pdf

# Initialise the Anthropic client once at module level. Streamlit re-imports
# this module on each run, but Python's module cache means this line only
# executes once per server process, so the client is effectively a singleton.
_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# System prompt injected into every follow-up chat API call to keep Claude
# focused on the paper in context and honest about what it doesn't know.
_SYSTEM_PROMPT = (
    "You are a research assistant helping a computational biologist "
    "understand scientific papers. Answer questions clearly and "
    "concisely based on the paper summary in the conversation. "
    "If asked something not covered in the summary, say so clearly."
)


def _build_pdf(pmid: str, summary: str) -> bytes:
    """
    Generates a styled PDF report containing the paper summary and returns it
    as raw bytes ready for a Streamlit download button.

    The function parses the markdown summary produced by Claude into labelled
    sections (title, authors, journal, key points, body), then lays them out
    using fpdf2 with a dark-navy/teal colour palette that matches the UI.

    Args:
        pmid (str): The PubMed ID shown in the PDF header, e.g. "37651234".
        summary (str): The markdown-formatted summary string produced by
                       run_agent(), e.g. "**Title:** ...\n**Authors:** ...".

    Returns:
        bytes: The complete PDF file as a bytes object. Pass directly to
               st.download_button's `data` parameter.
    """
    def clean(text: str) -> str:
        """
        Strips non-latin-1 characters so fpdf2 can encode them without errors.

        fpdf2's built-in fonts only support latin-1. NFKD normalisation decomposes
        accented characters into base + diacritic so the base survives the encode
        step; truly non-representable characters (e.g. CJK, emoji) are silently
        dropped by the 'ignore' error handler.

        Args:
            text (str): Any Unicode string.

        Returns:
            str: The text encoded to latin-1 and decoded back to a str, with
                 non-representable characters removed.
        """
        return unicodedata.normalize("NFKD", text).encode("latin-1", "ignore").decode("latin-1")

    def strip_md(text: str) -> str:
        """
        Removes common markdown syntax so the PDF contains plain prose.

        fpdf2 renders text literally, so bold markers (**) and heading hashes
        would appear as raw characters without this step.

        Args:
            text (str): A markdown-formatted string.

        Returns:
            str: The string with bold markers, heading hashes, and bullet
                 characters removed and surrounding whitespace stripped.
        """
        # Remove all sequences of asterisks (bold/italic markers).
        text = re.sub(r"\*+", "", text)
        # Remove markdown heading markers (## Heading → Heading).
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
        # Remove list bullet characters (- item or * item → item).
        text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)
        return text.strip()

    # ── Colour palette ───────────────────────────────────────────────────────
    # RGB tuples for every colour used in the PDF layout. Defined once here
    # so we can pass them to set_fill_color / set_text_color / set_draw_color.
    NAVY       = (15, 23, 42)    # dark header background
    TEAL       = (45, 212, 191)  # accent colour for labels and rules
    LIGHT_GREY = (248, 250, 252) # key-points block background
    BODY       = (51, 65, 85)    # main body text
    MUTED      = (148, 163, 184) # secondary/footer text
    BORDER     = (226, 232, 240) # horizontal rule colour
    WHITE      = (255, 255, 255) # text on dark backgrounds

    # ── Parse the markdown summary into structured fields ────────────────────
    # These variables are populated by scanning the summary line by line.
    # They are passed to the layout functions below once parsing is complete.
    title = ""
    authors = ""
    journal = ""
    key_points: list[str] = []  # bullet-point items become a visual sidebar
    body_lines: list[str] = []  # plain prose lines make up the summary body

    # Extract the title separately using a regex because it may span multiple
    # lines before the next ** marker. re.DOTALL makes . match newlines.
    m = re.search(r'\*\*Title:\*\*\s*(.*?)(?=\*\*)', summary, re.DOTALL)
    if m:
        title = strip_md(m.group(1).strip())

    # Walk the summary line by line to classify each into a field or the body.
    for line in summary.splitlines():
        s = line.strip()
        if not s:
            continue  # skip blank lines
        if re.match(r'\*\*Title:', s):
            continue  # already extracted via regex above; skip to avoid duplication
        elif re.match(r'\*\*Authors:\*\*', s):
            # Strip the label prefix, leaving just the author names.
            authors = strip_md(s.split('**Authors:**', 1)[1].strip())
        elif re.match(r'\*\*Journal:\*\*', s):
            # Strip the label prefix, leaving just the journal name/citation.
            journal = strip_md(s.split('**Journal:**', 1)[1].strip())
        elif re.match(r'^[-*]\s+', s):
            # Lines starting with a bullet marker become key-point entries.
            kp = strip_md(re.sub(r'^[-*]\s+', '', s))
            if kp:
                key_points.append(kp)
        elif not re.match(r'^(#+|\*\*)', s):
            # Lines that aren't headings or bold labels are plain body text.
            body_lines.append(strip_md(s))

    # Rejoin the prose lines and combine authors + journal into one block.
    body_text = "\n".join(body_lines)
    authors_journal = "\n".join(x for x in [authors, journal] if x)

    # ── Define the PDF class with a custom footer ────────────────────────────
    class ReportPDF(FPDF):
        def footer(self):
            """
            Renders the page footer on every page: a horizontal rule, a
            left-aligned "Generated by Research Assistant" label, and a
            right-aligned date stamp.
            """
            # Draw a thin horizontal rule 15 units from the bottom of the page.
            self.set_draw_color(*BORDER)
            self.set_line_width(0.3)
            self.line(0, self.h - 15, self.w, self.h - 15)
            # Place the footer text 12 units from the bottom.
            self.set_font("Helvetica", "", 9)
            self.set_text_color(*MUTED)
            self.set_xy(20, self.h - 12)
            self.cell(85, 5, "Generated by Research Assistant", align="L")
            self.set_xy(105, self.h - 12)
            self.cell(85, 5, date.today().strftime("%Y-%m-%d"), align="R")

    # ── Initialise the PDF document ──────────────────────────────────────────
    pdf = ReportPDF()
    # 20 mm left/right margins, 15 mm top margin.
    pdf.set_margins(20, 15, 20)
    # Automatically add a new page when content reaches 15 mm from the bottom.
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()

    # Pre-compute reusable layout measurements.
    w = pdf.w  # total page width in mm
    effective_width = w - 40  # width minus both 20 mm margins

    # ── Section 1: Header band ───────────────────────────────────────────────
    # Fill a 40 mm navy rectangle across the full page width as the header.
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, w, 40, style="F")

    # "RESEARCH ASSISTANT" sub-label in teal, top-left of the header band.
    pdf.set_xy(20, 11)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*TEAL)
    pdf.cell(0, 6, "RESEARCH ASSISTANT")

    # "Paper Summary" title in white, below the sub-label.
    pdf.set_xy(20, 19)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 8, "Paper Summary")

    # "PMID" label in muted grey, right-aligned at the same y as the sub-label.
    pdf.set_xy(20, 11)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*MUTED)
    pdf.cell(effective_width, 6, "PMID", align="R")

    # Actual PMID value in teal bold, right-aligned below the PMID label.
    pdf.set_xy(20, 19)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*TEAL)
    pdf.cell(effective_width, 8, clean(pmid), align="R")

    # ── Section 2: Teal accent rule ──────────────────────────────────────────
    # A 3 mm teal bar immediately below the navy header acts as a visual
    # separator between the header and the body content.
    pdf.set_fill_color(*TEAL)
    pdf.rect(0, 40, w, 3, style="F")

    # ── Body layout helpers ──────────────────────────────────────────────────
    # y tracks the current vertical cursor position in mm. We start 53 mm from
    # the top (40 mm header + 3 mm rule + 10 mm padding).
    y = 53

    def section_label(label: str) -> None:
        """
        Renders a small teal section heading (e.g. "TITLE", "SUMMARY") and
        advances the vertical cursor by 6 mm.

        Args:
            label (str): The label text. Will be upper-cased automatically
                         by the PDF cell's align logic — pass any case.
        """
        nonlocal y
        pdf.set_xy(20, y)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*TEAL)
        # label.upper() ensures consistent ALL-CAPS styling.
        pdf.cell(effective_width, 5, label.upper())
        y += 6  # advance cursor past the label

    def body_block(text: str) -> None:
        """
        Renders a block of body text in dark-body colour and updates y to the
        position below the last rendered line.

        Uses multi_cell so long lines wrap automatically within the page margins.

        Args:
            text (str): Plain (non-markdown) text to render.
        """
        nonlocal y
        pdf.set_xy(20, y)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(*BODY)
        # multi_cell wraps text and advances the internal cursor; we sync y
        # with get_y() afterwards so the next element is placed correctly.
        pdf.multi_cell(effective_width, 5, clean(text))
        y = pdf.get_y()

    def grey_rule() -> None:
        """
        Draws a thin light-grey horizontal rule across the text column and
        advances the cursor by 6 mm to create spacing between sections.
        """
        nonlocal y
        pdf.set_draw_color(*BORDER)
        pdf.set_line_width(0.3)
        # Draw the rule from the left margin to the right margin.
        pdf.line(20, y, w - 20, y)
        y += 6  # padding below the rule

    # ── Render title, authors/journal, and summary body ──────────────────────
    # Build a list of (label, text) pairs for the sections that have content.
    # Sections with empty text are omitted entirely.
    sections = []
    if title:
        sections.append(("TITLE", title))
    if authors_journal:
        sections.append(("AUTHORS & JOURNAL", authors_journal))
    if body_text:
        sections.append(("SUMMARY", body_text))

    # Render each section with a label, body, and a grey rule after it.
    # The grey rule is omitted after the last section only when there are no
    # key points; otherwise we always draw it to separate from the key points.
    for i, (label, text) in enumerate(sections):
        section_label(label)
        body_block(text)
        if i < len(sections) - 1 or key_points:
            y += 4  # small gap before the rule
            grey_rule()

    # ── Section 4: Key points block ──────────────────────────────────────────
    # Key points are rendered in a light-grey filled block with a 3 mm teal
    # left accent bar — a common design pattern for callout boxes.
    if key_points:
        section_label("KEY POINTS")
        y += 2  # extra padding between the label and the block background

        bx, bw = 20, effective_width  # block x position and width
        block_start = y  # remember where the block begins for the accent bar

        # Draw the top padding row of the grey background before the first item.
        pdf.set_fill_color(*LIGHT_GREY)
        pdf.set_xy(bx, y)
        pdf.cell(bw, 4, "", fill=True)
        y += 4

        # Render each key point as an indented bullet line inside the grey block.
        for kp in key_points:
            pdf.set_xy(bx, y)
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(*BODY)
            pdf.set_fill_color(*LIGHT_GREY)
            # Four spaces of indentation followed by a dash act as a visual
            # bullet point. fill=True keeps the grey background behind the text.
            pdf.multi_cell(bw, 5, clean(f"    - {kp}"), fill=True)
            y = pdf.get_y()

        # Draw the bottom padding row to close the grey block cleanly.
        pdf.set_fill_color(*LIGHT_GREY)
        pdf.set_xy(bx, y)
        pdf.cell(bw, 4, "", fill=True)
        y += 4

        # Draw the teal left accent bar spanning the full height of the block.
        # The bar is 3 mm wide and covers from block_start to y.
        pdf.set_fill_color(*TEAL)
        pdf.rect(bx, block_start, 3, y - block_start, style="F")

    # pdf.output() returns a bytearray; bytes() converts it for Streamlit.
    return bytes(pdf.output())


# ── UI ───────────────────────────────────────────────────────────────────────

# Inject custom CSS covering both PMID and PDF mode components.
# unsafe_allow_html=True is required to render raw HTML/CSS in Streamlit.
st.markdown("""
<style>
/* ── Shared layout components ────────────────────────────────────────────── */
.header-card {
    background: #1E293B;
    border-left: 4px solid #2DD4BF;
    padding: 1.2rem 1.5rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
}
.header-card h1 { color: #F1F5F9; margin: 0 0 0.25rem 0; font-size: 1.8rem; }
.header-card p  { color: #94A3B8; margin: 0; font-size: 0.95rem; }
.teal-divider   { border: none; border-top: 1px solid #2DD4BF; margin: 1.5rem 0; opacity: 0.5; }
.summary-card   { background: #1E293B; border: 1px solid #334155; border-radius: 8px;
                  padding: 1.5rem; color: #F1F5F9; line-height: 1.7; }

/* ── Mode selector cards ─────────────────────────────────────────────────── */
.mode-card {
    background: #1E293B;
    border: 1.5px solid #334155;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
}
.mode-card.active { border-color: #2DD4BF; }

/* ── Info card fields ────────────────────────────────────────────────────── */
.info-label {
    font-size: 11px;
    font-weight: 600;
    color: #2DD4BF;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.info-value { font-size: 13px; color: #CBD5E1; margin-bottom: 12px; }
.tag-pill {
    display: inline-block;
    background: #0F172A;
    border: 0.5px solid #334155;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    color: #94A3B8;
    margin: 2px 2px 0 0;
}
.finding-block {
    background: #0F172A;
    border-left: 2px solid #2DD4BF;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    font-size: 12px;
    color: #CBD5E1;
    margin-bottom: 6px;
    line-height: 1.5;
}
.limitation-block {
    background: #0F172A;
    border-left: 2px solid #F59E0B;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    font-size: 12px;
    color: #CBD5E1;
    margin-bottom: 6px;
    line-height: 1.5;
}
</style>

<div class="header-card">
    <h1>Research Assistant</h1>
    <p>AI-powered paper summarisation</p>
</div>
""", unsafe_allow_html=True)

# ── Session state initialisation ─────────────────────────────────────────────
# PMID mode: conversation history (index 0-1 are seed; 2+ are follow-ups)
if "messages" not in st.session_state:
    st.session_state.messages = []
# PDF mode keys are accessed via .get() with defaults throughout — no explicit
# initialisation needed, but listed here for documentation purposes:
#   pdf_summary, pdf_source, pdf_figures, pdf_info_card, pdf_bytes, pdf_messages

# ── Mode selector ─────────────────────────────────────────────────────────────
# Horizontal radio renders as two side-by-side options rather than a stacked list.
# label_visibility="collapsed" hides the label text while keeping accessibility.
mode = st.radio(
    "How would you like to add a paper?",
    ["Search by PMID", "Upload PDF"],
    horizontal=True,
    label_visibility="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# PMID MODE — fetch from PubMed / PMC (existing functionality, preserved intact)
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Search by PMID":

    pmid = st.text_input("PubMed ID")

    if st.button("Fetch & Summarise"):
        if not pmid.strip():
            st.warning("Please enter a PubMed ID.")
        else:
            with st.spinner("Fetching and summarising..."):
                summary, fetched_text = run_agent(pmid.strip())
                # Persist results so they survive Streamlit reruns triggered by
                # subsequent interactions (e.g. the user typing a follow-up).
                st.session_state["summary"] = summary
                st.session_state["pmid"] = pmid.strip()
                st.session_state["fetched_text"] = fetched_text
                # Seed the conversation with the original request and summary.
                # index 0 = user request, index 1 = initial summary.
                # Follow-up Q&A is appended from index 2 onward.
                st.session_state.messages = [
                    {"role": "user", "content": f"Fetch and summarise PMID {pmid.strip()}"},
                    {"role": "assistant", "content": summary},
                ]

    if "summary" in st.session_state:
        summary = st.session_state["summary"]
        saved_pmid = st.session_state["pmid"]

        st.markdown('<hr class="teal-divider">', unsafe_allow_html=True)

        # Render the summary inside a styled dark card; the markdown is already
        # produced by Claude so we embed it directly as HTML.
        st.markdown(f'<div class="summary-card">{summary}</div>', unsafe_allow_html=True)

        # Provenance caption — driven by the [Source: …] tag appended by
        # fetch_full_text() in research_agent.py.
        fetched_text = st.session_state.get("fetched_text", "")
        if "[Source: Full text" in fetched_text:
            st.caption("📄 Full text retrieved via PubMed Central")
        elif "[Source: Abstract only" in fetched_text:
            st.caption("📋 Abstract only — full text not available in PMC")

        st.markdown('<hr class="teal-divider">', unsafe_allow_html=True)

        # Download buttons rendered side by side in a two-column layout.
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download as Markdown",
                data=summary,
                file_name=f"{saved_pmid}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col2:
            # _build_pdf() is pure Python (no network) so calling it on every
            # rerun is cheap enough that we skip caching.
            st.download_button(
                label="Download as PDF",
                data=_build_pdf(saved_pmid, summary),
                file_name=f"{saved_pmid}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        st.markdown('<hr class="teal-divider">', unsafe_allow_html=True)

        # ── Follow-up chat ────────────────────────────────────────────────────
        # Render only follow-up turns (index 2+); the initial summary is already
        # shown in the styled card above.
        for msg in st.session_state.messages[2:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if question := st.chat_input("Ask a follow-up question about this paper..."):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Simple stateless completion — no tool use. The full messages
                    # list gives Claude complete context of the conversation so far.
                    api_response = _client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=1024,
                        system=_SYSTEM_PROMPT,
                        messages=st.session_state.messages,
                    )
                    # next() with default avoids StopIteration if no text block
                    answer = next(
                        (b.text for b in api_response.content if b.type == "text"),
                        "Unable to generate response.",
                    )
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# ══════════════════════════════════════════════════════════════════════════════
# PDF MODE — upload a paper PDF for full analysis with figure extraction
# ══════════════════════════════════════════════════════════════════════════════
else:
    uploaded_file = st.file_uploader(
        "Drop your paper PDF here",
        type="pdf",
        help="Upload any paper PDF for full text analysis including figures",
    )

    if uploaded_file:
        if st.button("Analyse paper", type="primary", use_container_width=True):
            pdf_bytes = uploaded_file.read()

            with st.spinner("Reading paper, extracting figures..."):
                summary, source_label, figures, info_card = summarise_pdf(pdf_bytes)

            # All PDF mode results are stored under separate session_state keys
            # so they never conflict with the PMID mode keys.
            st.session_state.pdf_summary = summary
            st.session_state.pdf_source = source_label
            st.session_state.pdf_figures = figures
            st.session_state.pdf_info_card = info_card
            st.session_state.pdf_bytes = pdf_bytes
            # Seed with the analysis request and initial summary.
            # index 0-1 are skipped when rendering the follow-up chat history.
            st.session_state.pdf_messages = [
                {"role": "user", "content": "Analyse this uploaded paper"},
                {"role": "assistant", "content": summary},
            ]

    # ── Results layout ────────────────────────────────────────────────────────
    # Shown persistently once a PDF has been analysed, regardless of whether
    # a new file is currently uploaded in the file uploader widget.
    if st.session_state.get("pdf_summary"):
        left_col, right_col = st.columns([3, 2])

        # ── LEFT COLUMN: narrative summary + follow-up chat ───────────────────
        with left_col:
            st.caption("📄 Full text via uploaded PDF")
            st.markdown("### Summary")
            st.markdown(st.session_state.pdf_summary)
            st.divider()
            st.markdown("### Ask a follow-up")

            # Render follow-up turns only — the seed entries at index 0-1 are
            # displayed above via st.markdown, not as chat bubbles.
            for msg in st.session_state.pdf_messages[2:]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if prompt := st.chat_input("Ask about this paper..."):
                st.session_state.pdf_messages.append({"role": "user", "content": prompt})

                # Re-encode the PDF for this API call so the model always has the
                # full paper in context, regardless of how many turns have passed.
                pdf_b64 = base64.standard_b64encode(
                    st.session_state.pdf_bytes
                ).decode("utf-8")

                # Build the messages list: all previous turns as plain text, then
                # the current prompt with the PDF attached as a document block.
                api_messages = []
                for msg in st.session_state.pdf_messages[:-1]:
                    api_messages.append(msg)
                api_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                })

                chat_response = _client.messages.create(
                    model="claude-opus-4-5",
                    max_tokens=1000,
                    system="You are a research assistant. Answer questions about this paper precisely.",
                    messages=api_messages,
                )

                answer = next(
                    (b.text for b in chat_response.content if b.type == "text"),
                    "Unable to generate response.",
                )

                st.session_state.pdf_messages.append({"role": "assistant", "content": answer})
                st.rerun()

        # ── RIGHT COLUMN: structured info card + figure gallery ───────────────
        with right_col:

            # ── INFO CARD ─────────────────────────────────────────────────────
            if st.session_state.get("pdf_info_card"):
                card = st.session_state.pdf_info_card
                st.markdown("### Paper at a glance")

                if card.get("hypothesis"):
                    st.markdown(
                        f'<div class="info-label">Hypothesis</div>'
                        f'<div class="info-value">{card["hypothesis"]}</div>',
                        unsafe_allow_html=True,
                    )

                # Model system and sample size rendered side by side
                m1, m2 = st.columns(2)
                with m1:
                    if card.get("model_system"):
                        st.markdown(
                            f'<div class="info-label">Model system</div>'
                            f'<div class="info-value">{card["model_system"]}</div>',
                            unsafe_allow_html=True,
                        )
                with m2:
                    if card.get("sample_size"):
                        st.markdown(
                            f'<div class="info-label">Sample size</div>'
                            f'<div class="info-value">{card["sample_size"]}</div>',
                            unsafe_allow_html=True,
                        )

                # Statistical methods as small pill badges
                if card.get("statistical_tests"):
                    pills_html = "".join(
                        f'<span class="tag-pill">{t}</span>'
                        for t in card["statistical_tests"]
                    )
                    st.markdown(
                        f'<div class="info-label">Statistical methods</div>'
                        f'<div style="margin-bottom:12px">{pills_html}</div>',
                        unsafe_allow_html=True,
                    )

                # Datasets and tools as pill badges
                if card.get("datasets_tools"):
                    pills_html = "".join(
                        f'<span class="tag-pill">{d}</span>'
                        for d in card["datasets_tools"]
                    )
                    st.markdown(
                        f'<div class="info-label">Datasets & tools</div>'
                        f'<div style="margin-bottom:12px">{pills_html}</div>',
                        unsafe_allow_html=True,
                    )

                # Key findings with a teal left-border accent
                if card.get("key_findings"):
                    findings_html = "".join(
                        f'<div class="finding-block">{f}</div>'
                        for f in card["key_findings"]
                    )
                    st.markdown(
                        f'<div class="info-label">Key findings</div>{findings_html}',
                        unsafe_allow_html=True,
                    )

                # Limitations with an amber left-border accent
                if card.get("limitations"):
                    lims_html = "".join(
                        f'<div class="limitation-block">{l}</div>'
                        for l in card["limitations"]
                    )
                    st.markdown(
                        f'<div class="info-label">Limitations</div>{lims_html}',
                        unsafe_allow_html=True,
                    )

                if card.get("relevance_to_drug_discovery"):
                    st.markdown(
                        f'<div class="info-label">Drug discovery relevance</div>'
                        f'<div class="info-value">{card["relevance_to_drug_discovery"]}</div>',
                        unsafe_allow_html=True,
                    )

            # ── FIGURE GALLERY ────────────────────────────────────────────────
            # Two-column grid; pairs of figures fill each row. Figures are stored
            # as raw bytes in session state so we pass them directly to st.image().
            figures = st.session_state.get("pdf_figures", [])
            if figures:
                st.markdown(f"### Figures ({len(figures)} extracted)")

                for i in range(0, len(figures), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(figures):
                            fig = figures[idx]
                            with col:
                                st.image(
                                    fig["bytes"],
                                    caption=f"Figure {idx + 1}",
                                    use_container_width=True,
                                )
                                with st.expander("Claude's interpretation"):
                                    st.markdown(fig["interpretation"])
            else:
                st.caption(
                    "No extractable figures found — "
                    "figures may be vector graphics."
                )
