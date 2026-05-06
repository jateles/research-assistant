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
from research_agent import run_agent, summarise_pdf, get_pmcid
from landscape_agent import anchor_builder, literature_scout, relevance_ranker

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


# @st.dialog creates a modal overlay that appears when the decorated function
# is called. width="large" uses the wider modal variant. The function body runs
# like a normal Streamlit script inside the modal — st.image, st.markdown, etc.
# all work as usual. The modal closes when the user clicks outside it or presses
# Escape. The decorator must be applied at definition time, before any UI code
# calls the function.
@st.dialog("Figure detail", width="large")
def show_figure_popup(fig: dict, fig_number: int) -> None:
    """
    Displays a figure in a large popup modal with Claude's interpretation.

    Args:
        fig (dict): Figure dict with 'bytes' (raw image bytes) and
                    'interpretation' (Claude's description string) keys.
        fig_number (int): 1-based figure number shown as the image caption.
    """
    st.image(
        fig["bytes"],
        caption=f"Figure {fig_number}",
        use_container_width=True,
    )
    st.divider()
    st.markdown("**Claude's interpretation**")
    st.markdown(fig["interpretation"])


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

# ── Sidebar navigation ────────────────────────────────────────────────────────
# The sidebar radio persists across all modes and survives reruns. Using the
# sidebar instead of a horizontal in-page radio keeps the main content area
# uncluttered, especially important for the landscape mode's multi-step flow.
with st.sidebar:
    st.markdown("### Research Assistant")
    st.markdown("---")
    # label_visibility="collapsed" hides the widget label while keeping the
    # radio accessible — the "### Research Assistant" heading acts as the visual label.
    mode = st.radio(
        "Mode",
        ["Single paper", "Upload PDF", "Literature landscape"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Built with Claude + PubMed")

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE PAPER MODE — fetch from PubMed / PMC (existing functionality intact)
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Single paper":

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
elif mode == "Upload PDF":
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
            # Two-column grid sorted by image area (largest first, done in
            # research_agent.py). Figures are stored as raw bytes in session
            # state and passed directly to st.image(). Clicking "View & interpret"
            # opens the figure in a large @st.dialog modal overlay.
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
                                # Unique key required — Streamlit errors if two
                                # buttons in the same script share the same key.
                                if st.button(
                                    "🔍 View & interpret",
                                    key=f"fig_btn_{idx}",
                                    use_container_width=True,
                                ):
                                    show_figure_popup(fig, idx + 1)
            else:
                st.caption(
                    "No extractable figures found — "
                    "figures may be vector graphics."
                )

# ══════════════════════════════════════════════════════════════════════════════
# LITERATURE LANDSCAPE MODE — multi-agent pipeline (Steps 1-2 active)
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Literature landscape":

    # ── Session state initialisation ─────────────────────────────────────────
    # landscape_step controls which screen is shown:
    #   1 = user input form
    #   2 = anchor review and edit
    #   3 = paper curation (Session 2)
    #   4 = landscape report (Session 3)
    # Initialised here rather than at module level so it only exists when the
    # user is actually in this mode.
    if "landscape_step" not in st.session_state:
        st.session_state.landscape_step = 1

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1: User input
    # ──────────────────────────────────────────────────────────────────────────
    if st.session_state.landscape_step == 1:

        st.markdown("## Literature landscape")
        st.markdown(
            "Enter a topic, research question, or PMID. "
            "Claude will build a search anchor for you to review."
        )

        # Primary input — accepts free text, a PMID, or a research question.
        # anchor_builder's regex will automatically detect any 7-8 digit PMID.
        user_input = st.text_area(
            "What do you want to explore?",
            height=100,
            placeholder=(
                "e.g. 'Stochastic models of haematopoietic commitment' "
                "or 'PMID 23990771' "
                "or 'Does transcriptional noise drive cell fate decisions?'"
            ),
        )

        # Optional context lets the user bias the anchor toward their specific
        # use case without changing the core topic string.
        user_context = st.text_input(
            "Optional: your role or intent (helps focus the anchor)",
            placeholder="e.g. 'Evaluating methods for my cancer genomics dataset'",
        )

        # disabled=not user_input.strip() prevents submitting an empty form —
        # Streamlit evaluates this expression on every rerun so the button
        # enables/disables reactively as the user types.
        if st.button(
            "Build anchor",
            type="primary",
            use_container_width=True,
            disabled=not user_input.strip(),
        ):
            # Concatenate topic and optional context into a single string so
            # anchor_builder sees full intent in one pass.
            full_input = user_input.strip()
            if user_context.strip():
                full_input += f"\n\nContext: {user_context.strip()}"

            with st.spinner("Agent 0 — building your research anchor..."):
                anchor = anchor_builder(full_input)

            # Persist anchor and raw input separately — the raw input is shown
            # in the Step 2 expander so the user can see what they originally typed.
            st.session_state.landscape_anchor = anchor
            st.session_state.landscape_input = user_input.strip()
            st.session_state.landscape_step = 2
            st.rerun()

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2: Anchor review and edit
    # ──────────────────────────────────────────────────────────────────────────
    elif st.session_state.landscape_step == 2:

        anchor = st.session_state.landscape_anchor

        st.markdown("## Review your anchor")
        st.caption(
            "Agent 0 proposed this research scope. "
            "Edit any field before searching."
        )

        # Show the original input so the user can cross-check Agent 0's
        # interpretation against what they actually typed.
        with st.expander("You entered", expanded=False):
            st.markdown(f"_{st.session_state.landscape_input}_")

        st.markdown("---")

        # ── Editable anchor fields ────────────────────────────────────────────
        # All fields are pre-populated from the anchor dict returned by Agent 0.
        # The user can freely edit before confirming; the confirmed values (not
        # Agent 0's originals) are what downstream agents receive.

        # Central question — the primary filter for relevance ranking.
        core_question = st.text_area(
            "Core question",
            value=anchor.get("core_question", ""),
            height=80,
            help="The central question your landscape will answer",
        )

        # Field scope — defines the search perimeter and time range.
        field_scope = st.text_area(
            "Field scope",
            value=anchor.get("field_scope", ""),
            height=80,
            help="Field boundaries and time range for the search",
        )

        # Key debates displayed as a comma-separated string — simpler than a
        # dynamic tag widget while preserving all the same data for downstream use.
        debates_list = anchor.get("key_debates", [])
        debates_str = st.text_input(
            "Key debates (comma separated — edit or add your own)",
            value=", ".join(debates_list),
        )

        st.markdown("**Search strategy**")

        strategy = anchor.get("search_strategy", {})

        # Three side-by-side checkboxes — pre-set by Agent 0, user can toggle.
        # Citation traversal is disabled (greyed out) when no PMID was detected
        # because the scout has no seed paper to traverse from.
        col1, col2, col3 = st.columns(3)
        with col1:
            keyword = st.checkbox(
                "Keyword search",
                value=strategy.get("keyword_search", True),
                help="Search PubMed using terms from the core question",
            )
        with col2:
            citation = st.checkbox(
                "Citation traversal",
                value=strategy.get("citation_traversal", False),
                help=(
                    "Fetch papers that cite or are cited by a given paper. "
                    "Requires a PMID in the input."
                ),
                # Disable the checkbox entirely if no PMID was found —
                # the scout can't traverse citations without a seed paper.
                disabled=not anchor.get("detected_pmid"),
            )
        with col3:
            author_net = st.checkbox(
                "Author network",
                value=strategy.get("author_network", True),
                help="Fetch recent papers by key authors in this area",
            )

        # Inform the user when a PMID was detected so they understand why
        # citation traversal is available (or unavailable).
        if anchor.get("detected_pmid"):
            st.caption(
                f"PMID {anchor['detected_pmid']} detected — "
                "citation traversal is available"
            )

        st.markdown("---")

        # ── Action buttons ────────────────────────────────────────────────────
        # Three buttons in a 1:1:2 column layout:
        #   ← Back         — return to Step 1 and clear the anchor
        #   Regenerate ↺   — re-run Agent 0 with the same input
        #   Confirm →      — save edited anchor and advance to Step 3
        col_back, col_regen, col_confirm = st.columns([1, 1, 2])

        with col_back:
            if st.button("← Back", use_container_width=True):
                # Clear the anchor from session state so Step 1 starts fresh.
                st.session_state.landscape_step = 1
                if "landscape_anchor" in st.session_state:
                    del st.session_state.landscape_anchor
                st.rerun()

        with col_regen:
            if st.button("Regenerate ↺", use_container_width=True):
                # Re-run anchor_builder with the same raw input — useful if
                # Agent 0's first proposal missed the intent.
                with st.spinner("Regenerating anchor..."):
                    new_anchor = anchor_builder(
                        st.session_state.landscape_input
                    )
                st.session_state.landscape_anchor = new_anchor
                st.rerun()

        with col_confirm:
            if st.button(
                "Confirm anchor and search →",
                type="primary",
                use_container_width=True,
            ):
                # Build the confirmed anchor from the current widget values —
                # this overwrites Agent 0's originals with any edits the user made.
                edited_anchor = {
                    "core_question": core_question,
                    "field_scope": field_scope,
                    # Split the comma-separated string back into a list,
                    # stripping whitespace and dropping any empty items.
                    "key_debates": [
                        d.strip()
                        for d in debates_str.split(",")
                        if d.strip()
                    ],
                    "search_strategy": {
                        "keyword_search": keyword,
                        "citation_traversal": citation,
                        "author_network": author_net,
                    },
                    # Preserve the detected PMID so the scout can use it even
                    # if the user didn't change the citation_traversal flag.
                    "detected_pmid": anchor.get("detected_pmid"),
                }

                st.session_state.landscape_anchor = edited_anchor
                st.session_state.landscape_step = 3
                st.rerun()

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3: Paper curation screen
    # Agent 1 (literature_scout) and Agent 2 (relevance_ranker) run here.
    # Results are cached in session state so they survive Streamlit reruns.
    # ──────────────────────────────────────────────────────────────────────────
    elif st.session_state.landscape_step == 3:

        anchor = st.session_state.landscape_anchor

        # Run scout and ranker only once per anchor — results are stored in
        # session state and reused on every subsequent rerun of this step.
        if "landscape_candidates" not in st.session_state:

            with st.status(
                "Searching literature...",
                expanded=True
            ) as status:

                st.write(
                    "Querying Semantic Scholar and PubMed — "
                    "this takes 20-30 seconds..."
                )

                # Agent 1: fetch candidate papers from all enabled sources.
                candidates = literature_scout(anchor)
                st.write(f"Found {len(candidates)} candidate papers")

                # Agent 2: score and sort candidates by relevance to anchor.
                st.write("Ranking by relevance to your anchor...")
                ranked = relevance_ranker(candidates, anchor)
                st.write("Ranking complete")

                # Deferred full-text check — only verify the top 20 ranked
                # papers rather than all candidates to keep latency acceptable.
                st.write("Checking full text availability...")
                from landscape_agent import check_full_text_batch
                ranked_checked = check_full_text_batch(ranked[:20])
                for i, paper in enumerate(ranked_checked):
                    ranked[i] = paper

                status.update(
                    label=f"Found {len(ranked)} relevant papers",
                    state="complete"
                )

            # Persist results so this block doesn't re-execute on rerun.
            st.session_state.landscape_candidates = ranked

            # Pre-select the top 5 papers by default; user can change this.
            top_pmids = {p["pmid"] for p in ranked[:5]}
            st.session_state.selected_pmids = top_pmids
            st.session_state.manual_papers = []
            st.session_state.paper_pdfs = {}

        # Retrieve the stored candidate list for rendering below.
        candidates = st.session_state.landscape_candidates

        st.markdown("## Select papers for your landscape")
        st.caption(
            "Agent 2 has ranked these papers by relevance. "
            "Select which to include. Add your own PMIDs below."
        )

        # Collapsible reminder of the confirmed anchor so the user can
        # cross-check paper relevance against their original scope.
        with st.expander("Your anchor", expanded=False):
            st.markdown(
                f"**Core question:** {anchor.get('core_question', '')}"
            )
            st.markdown(
                f"**Scope:** {anchor.get('field_scope', '')}"
            )

        st.markdown("---")

        # Two-column layout: paper list on the left (wider),
        # selection summary and action buttons on the right.
        left_col, right_col = st.columns([3, 2])

        with left_col:

            st.markdown("### Ranked papers")

            # Human-readable badge for each retrieval relationship type.
            badge_labels = {
                "cited_by_anchor": "🟣 Cites this paper",
                "cites_anchor":    "🔵 Cited by this paper",
                "same_author":     "🟡 Same author",
                "keyword_match":   "⚪ Keyword match",
            }

            # ── Section 1: Top 20 papers — always visible ─────────────────
            for paper in candidates[:20]:
                pmid = paper["pmid"]

                with st.container(border=True):

                    # Top row: checkbox on the left, title and authors on the right.
                    col_check, col_title = st.columns([1, 8])

                    with col_check:
                        # Read current selection state from the set; write back
                        # after the checkbox renders so the set stays in sync.
                        is_selected = pmid in st.session_state.selected_pmids
                        checked = st.checkbox(
                            "Select",
                            value=is_selected,
                            key=f"check_{pmid}",
                            label_visibility="collapsed",
                        )
                        if checked:
                            st.session_state.selected_pmids.add(pmid)
                        else:
                            st.session_state.selected_pmids.discard(pmid)

                    with col_title:
                        st.markdown(f"**{paper['title']}**")
                        st.caption(
                            f"{paper['authors']} · "
                            f"{paper['journal'] or 'Unknown journal'} · "
                            f"{paper['year'] or 'Unknown year'}"
                        )

                    # Bottom row: relationship badge, relevance score bar,
                    # full-text status with optional inline PDF upload.
                    col_badge, col_score, col_ft = st.columns([3, 2, 3])

                    with col_badge:
                        st.caption(
                            badge_labels.get(paper["relationship"], "⚪ Found")
                        )
                        if paper.get("intent"):
                            st.caption(f"Citation intent: {paper['intent']}")

                    with col_score:
                        score = paper.get("relevance_score", 0)
                        st.caption(f"Relevance: {score}/100")
                        st.progress(score / 100)

                    with col_ft:
                        status = paper.get("full_text_status", "unknown")

                        if status == "available":
                            st.caption("📄 Full text available")
                            upload_label = "Upload PDF for richer analysis"
                        elif status == "unavailable":
                            st.caption("📋 Abstract only")
                            upload_label = "Upload PDF"
                        else:
                            st.caption(
                                "📋 Full text status unknown — "
                                "upload PDF if available"
                            )
                            upload_label = "Upload PDF"

                        uploaded_pdf = st.file_uploader(
                            upload_label,
                            type="pdf",
                            key=f"pdf_{pmid}",
                            label_visibility="collapsed",
                        )
                        if uploaded_pdf:
                            pdf_bytes = uploaded_pdf.read()
                            if pdf_bytes:
                                st.session_state.paper_pdfs[pmid] = pdf_bytes
                                st.caption("✓ PDF uploaded")

            # ── Section 2: Remaining papers — collapsed expander ───────────
            remaining = candidates[20:]
            if remaining:
                with st.expander(
                    f"Show {len(remaining)} more papers",
                    expanded=False,
                ):
                    for paper in remaining:
                        pmid = paper["pmid"]

                        with st.container(border=True):

                            col_check, col_title = st.columns([1, 8])

                            with col_check:
                                is_selected = pmid in st.session_state.selected_pmids
                                checked = st.checkbox(
                                    "Select",
                                    value=is_selected,
                                    key=f"check_{pmid}",
                                    label_visibility="collapsed",
                                )
                                if checked:
                                    st.session_state.selected_pmids.add(pmid)
                                else:
                                    st.session_state.selected_pmids.discard(pmid)

                            with col_title:
                                st.markdown(f"**{paper['title']}**")
                                st.caption(
                                    f"{paper['authors']} · "
                                    f"{paper['journal'] or 'Unknown journal'} · "
                                    f"{paper['year'] or 'Unknown year'}"
                                )

                            col_badge, col_score, col_ft = st.columns([3, 2, 3])

                            with col_badge:
                                st.caption(
                                    badge_labels.get(paper["relationship"], "⚪ Found")
                                )
                                if paper.get("intent"):
                                    st.caption(f"Citation intent: {paper['intent']}")

                            with col_score:
                                score = paper.get("relevance_score", 0)
                                st.caption(f"Relevance: {score}/100")
                                st.progress(score / 100)

                            with col_ft:
                                status = paper.get("full_text_status", "unknown")

                                if status == "available":
                                    st.caption("📄 Full text available")
                                    upload_label = "Upload PDF for richer analysis"
                                elif status == "unavailable":
                                    st.caption("📋 Abstract only")
                                    upload_label = "Upload PDF"
                                else:
                                    st.caption(
                                        "📋 Full text status unknown — "
                                        "upload PDF if available"
                                    )
                                    upload_label = "Upload PDF"

                                uploaded_pdf = st.file_uploader(
                                    upload_label,
                                    type="pdf",
                                    key=f"pdf_{pmid}",
                                    label_visibility="collapsed",
                                )
                                if uploaded_pdf:
                                    pdf_bytes = uploaded_pdf.read()
                                    if pdf_bytes:
                                        st.session_state.paper_pdfs[pmid] = pdf_bytes
                                        st.caption("✓ PDF uploaded")

            # ── Manually added papers ─────────────────────────────────────
            # Rendered below the ranked list once any PMIDs have been resolved
            # via the "Resolve and add papers" button.
            manual_papers = st.session_state.get("manual_papers", [])
            if manual_papers:
                st.markdown("### Added manually")
                for paper in manual_papers:
                    pmid = paper["pmid"]
                    with st.container(border=True):
                        col_check, col_title = st.columns([1, 8])
                        with col_check:
                            checked = st.checkbox(
                                "Select",
                                value=pmid in st.session_state.selected_pmids,
                                key=f"check_manual_{pmid}",
                                label_visibility="collapsed",
                            )
                            if checked:
                                st.session_state.selected_pmids.add(pmid)
                            else:
                                st.session_state.selected_pmids.discard(pmid)
                        with col_title:
                            st.markdown(f"**{paper['title']}**")
                            st.caption(
                                f"{paper.get('authors', '')} · Added manually"
                            )
                        status = paper.get("full_text_status", "unknown")

                        if status == "available":
                            st.caption("📄 Full text available")
                            upload_label = "Upload PDF for richer analysis"
                        elif status == "unavailable":
                            st.caption("📋 Abstract only")
                            upload_label = "Upload PDF"
                        else:
                            st.caption(
                                "📋 Full text status unknown — "
                                "upload PDF if available"
                            )
                            upload_label = "Upload PDF"

                        uploaded_pdf = st.file_uploader(
                            upload_label,
                            type="pdf",
                            key=f"pdf_manual_{pmid}",
                            label_visibility="collapsed",
                        )
                        if uploaded_pdf:
                            pdf_bytes = uploaded_pdf.read()
                            if pdf_bytes:
                                st.session_state.paper_pdfs[pmid] = pdf_bytes
                                st.caption("✓ PDF uploaded")

            # ── Add your own papers ───────────────────────────────────────
            st.markdown("---")
            st.markdown("### Add your own papers")
            st.caption("Paste PMIDs one per line or comma separated")

            manual_input = st.text_area(
                "PMIDs to add",
                height=80,
                placeholder="23990771\n26040267\n28212749",
                label_visibility="collapsed",
            )

            if st.button("Resolve and add papers", use_container_width=True):
                if manual_input.strip():
                    # Accept newlines, commas, or whitespace as separators.
                    raw_pmids = re.split(r'[\n,\s]+', manual_input.strip())
                    # Keep only 7-8 digit strings — anything else is not a PMID.
                    valid_pmids = [
                        p.strip() for p in raw_pmids
                        if re.match(r'^\d{7,8}$', p.strip())
                    ]

                    if valid_pmids:
                        with st.spinner(
                            f"Resolving {len(valid_pmids)} papers..."
                        ):
                            from research_agent import fetch_abstract
                            new_papers = []
                            # Build lookup set once to avoid O(n²) membership checks.
                            existing_pmids = {p["pmid"] for p in candidates}

                            for pmid in valid_pmids:
                                if pmid in existing_pmids:
                                    st.caption(f"PMID {pmid} already in list")
                                    continue

                                try:
                                    abstract = fetch_abstract(pmid)
                                    # Extract title: first non-empty line over 20
                                    # chars that doesn't start with a digit.
                                    lines = [
                                        l.strip()
                                        for l in abstract.split("\n")
                                        if l.strip()
                                    ]
                                    title = next(
                                        (
                                            l for l in lines
                                            if len(l) > 20
                                            and not l[0].isdigit()
                                        ),
                                        f"PMID {pmid}",
                                    )
                                    pmcid = get_pmcid(pmid)
                                    new_papers.append({
                                        "pmid": pmid,
                                        "title": title,
                                        "authors": "",
                                        "journal": "",
                                        "year": 0,
                                        "relationship": "manual",
                                        "intent": "",
                                        "has_full_text": pmcid is not None,
                                        "pmcid": pmcid,
                                        "relevance_score": 100,
                                        "relevance_rationale": "Added manually",
                                    })
                                    # Auto-select every manually added paper.
                                    st.session_state.selected_pmids.add(pmid)
                                except Exception as e:
                                    st.warning(
                                        f"Could not resolve PMID {pmid}: {e}"
                                    )

                        if new_papers:
                            st.session_state.manual_papers.extend(new_papers)
                            st.success(f"Added {len(new_papers)} papers")
                            st.rerun()
                    else:
                        st.warning(
                            "No valid PMIDs found. "
                            "PMIDs are 7-8 digit numbers."
                        )

        with right_col:

            st.markdown("### Your selection")

            # Count selected papers split by source for the metrics display.
            n_selected = len(st.session_state.selected_pmids)
            n_ranked = len([
                p for p in candidates
                if p["pmid"] in st.session_state.selected_pmids
            ])
            n_manual = len([
                p for p in st.session_state.get("manual_papers", [])
                if p["pmid"] in st.session_state.selected_pmids
            ])

            # Summary metrics card.
            with st.container(border=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Selected", n_selected)
                with col2:
                    st.metric("From ranked", n_ranked)
                if n_manual > 0:
                    st.metric("Added manually", n_manual)

            st.markdown("---")

            # Disable the generate button until at least 2 papers are selected —
            # Agent 3 needs a minimum of 2 to produce a meaningful landscape.
            generate_disabled = n_selected < 2
            if generate_disabled:
                st.caption(
                    "Select at least 2 papers to generate your landscape"
                )

            if st.button(
                "Generate landscape →",
                type="primary",
                use_container_width=True,
                disabled=generate_disabled,
            ):
                st.session_state.landscape_step = 4
                st.rerun()

            if st.button("Use all ranked papers", use_container_width=True):
                # Select every candidate from the ranked list in one click.
                st.session_state.selected_pmids = {
                    p["pmid"] for p in candidates
                }
                st.rerun()

            st.markdown("---")
            st.markdown("**What Agent 3 will receive:**")
            st.caption(
                f"• {n_selected} selected papers\n"
                f"• Your confirmed anchor document\n"
                f"• Full text where available, abstracts otherwise"
            )

            # Back button clears cached candidates so the scout and ranker
            # re-run if the user edits the anchor and returns to Step 3.
            st.markdown("---")
            if st.button("← Back to anchor", use_container_width=True):
                if "landscape_candidates" in st.session_state:
                    del st.session_state.landscape_candidates
                st.session_state.landscape_step = 2
                st.rerun()

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4: Placeholder — Synthesis Agent coming in Session 3
    # ──────────────────────────────────────────────────────────────────────────
    elif st.session_state.landscape_step == 4:
        st.info("Synthesis Agent coming in Session 3.")
        if st.button("← Back to paper selection"):
            st.session_state.landscape_step = 3
            st.rerun()
