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
from datetime import date
import os
import streamlit as st
import anthropic
from fpdf import FPDF
from research_agent import run_agent

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

# Inject custom CSS for the header card, dividers, and summary card.
# unsafe_allow_html=True is required to render raw HTML/CSS in Streamlit.
# The styles match the PDF colour palette for a consistent look across formats.
st.markdown("""
<style>
.header-card {
    background: #1E293B;
    border-left: 4px solid #2DD4BF;
    padding: 1.2rem 1.5rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
}
.header-card h1 {
    color: #F1F5F9;
    margin: 0 0 0.25rem 0;
    font-size: 1.8rem;
}
.header-card p {
    color: #94A3B8;
    margin: 0;
    font-size: 0.95rem;
}
.teal-divider {
    border: none;
    border-top: 1px solid #2DD4BF;
    margin: 1.5rem 0;
    opacity: 0.5;
}
.summary-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 1.5rem;
    color: #F1F5F9;
    line-height: 1.7;
}
</style>

<div class="header-card">
    <h1>Research Assistant</h1>
    <p>AI-powered paper summarisation</p>
</div>
""", unsafe_allow_html=True)

# ── Session state initialisation ─────────────────────────────────────────────
#
# st.session_state.messages stores the conversation history as a list of
# {"role": ..., "content": ...} dicts in the format expected by the Claude API.
# It needs to persist across reruns because Streamlit re-executes the entire
# script on every user interaction; without session state, the history would
# be lost and Claude would lose context between follow-up questions.
# The first two entries (index 0 and 1) are always the original fetch request
# and the initial summary; subsequent entries are follow-up Q&A turns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# st.text_input renders a single-line text field and returns the current value
# as a string. The empty string "" is returned until the user types something.
pmid = st.text_input("PubMed ID")

# st.button renders a clickable button and returns True only on the specific
# rerun triggered by that button click; it returns False on all other reruns.
if st.button("Fetch & Summarise"):
    if not pmid.strip():
        # st.warning renders a yellow warning box visible to the user.
        st.warning("Please enter a PubMed ID.")
    else:
        # st.spinner shows an animated loading indicator for the duration of
        # the with block, giving the user feedback while run_agent() runs.
        with st.spinner("Fetching and summarising..."):
            summary, fetched_text = run_agent(pmid.strip())

            # Store the summary in session state so it survives the next rerun
            # (Streamlit reruns after every button click). Without this, the
            # summary would disappear as soon as the user interacts with the page.
            st.session_state["summary"] = summary

            # Store the PMID so the download file names remain correct even if
            # the user edits the text input after the fetch has completed.
            st.session_state["pmid"] = pmid.strip()

            # Store the raw fetched text so we can show the provenance caption
            # below the summary card without re-fetching.
            st.session_state["fetched_text"] = fetched_text

            # Seed the messages list with the original request and summary as
            # the first two turns. Follow-up questions will be appended here.
            # These first two entries are skipped when rendering the chat history
            # below (we use messages[2:]) because the summary is already shown
            # in the styled summary card above the chat area.
            st.session_state.messages = [
                {"role": "user", "content": f"Fetch and summarise PMID {pmid.strip()}"},
                {"role": "assistant", "content": summary},
            ]

# ── Render summary and download buttons if a summary has been fetched ─────────
#
# st.session_state["summary"] is set by the button handler above. Checking for
# it here (rather than inside the button block) means the summary persists
# across reruns even when the user is typing a follow-up question rather than
# clicking the button again.
if "summary" in st.session_state:
    # Retrieve persisted values from session state.
    summary = st.session_state["summary"]
    saved_pmid = st.session_state["pmid"]

    # Render a thin teal horizontal rule as a visual separator.
    st.markdown('<hr class="teal-divider">', unsafe_allow_html=True)

    # Render the summary inside a styled dark card.
    # The summary is already markdown-formatted by Claude, so we embed it
    # directly as HTML inside the card div; unsafe_allow_html is needed.
    st.markdown(
        f'<div class="summary-card">{summary}</div>',
        unsafe_allow_html=True,
    )

    # Show a provenance caption indicating whether full text or abstract was
    # used. The tag appended by fetch_full_text() in research_agent.py is the
    # signal — no tag means something went wrong or the field is empty.
    fetched_text = st.session_state.get("fetched_text", "")
    if "[Source: Full text" in fetched_text:
        st.caption("📄 Full text retrieved via PubMed Central")
    elif "[Source: Abstract only" in fetched_text:
        st.caption("📋 Abstract only — full text not available in PMC")

    st.markdown('<hr class="teal-divider">', unsafe_allow_html=True)

    # st.columns(2) returns two equal-width column objects. Placing each
    # download button in its own column renders them side by side rather than
    # stacked vertically.
    col1, col2 = st.columns(2)

    with col1:
        # st.download_button renders a button that triggers a file download in
        # the user's browser when clicked. Returns True on the click rerun.
        # data accepts str or bytes; mime tells the browser what type to expect.
        st.download_button(
            label="Download as Markdown",
            data=summary,           # summary is already a str; no conversion needed
            file_name=f"{saved_pmid}.md",
            mime="text/markdown",
            use_container_width=True,  # stretch to fill the column width
        )

    with col2:
        # _build_pdf() is called on every rerun where a summary exists. The
        # call is cheap enough (pure Python, no network) that we don't cache it.
        st.download_button(
            label="Download as PDF",
            data=_build_pdf(saved_pmid, summary),  # returns bytes
            file_name=f"{saved_pmid}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown('<hr class="teal-divider">', unsafe_allow_html=True)

    # ── Follow-up chat loop ──────────────────────────────────────────────────
    # Render only the follow-up turns (index 2 onwards); the initial fetch
    # request and summary are shown in the styled card above, not here.
    for msg in st.session_state.messages[2:]:
        # st.chat_message renders a chat bubble with a role avatar (user/assistant).
        with st.chat_message(msg["role"]):
            # st.markdown renders the message content as formatted markdown.
            st.markdown(msg["content"])

    # st.chat_input renders a fixed chat input bar at the bottom of the page.
    # It returns the submitted string when the user presses Enter or the send
    # button, and None on all other reruns. The walrus operator (:=) assigns
    # the value and enters the if block only when a non-None string is returned.
    if question := st.chat_input("Ask a follow-up question about this paper..."):
        # Append the new user question to the messages list before calling the
        # API so the API receives the complete conversation history including
        # this new turn. Without this append, the history would be one turn
        # behind and Claude would not see the question it is answering.
        st.session_state.messages.append({"role": "user", "content": question})

        # Display the user's question immediately in a chat bubble so the UI
        # feels responsive while we wait for the API response.
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # ── Follow-up API call ────────────────────────────────────────
                #
                # This is a simple stateless completion call — no tool use.
                # The messages list at this point contains:
                #   [0] user: original fetch request
                #   [1] assistant: initial summary
                #   [2..n] alternating user/assistant follow-up turns
                #   [last] user: the question just entered above
                #
                # We pass the full messages list so Claude has complete context
                # of everything discussed so far about this paper.
                #
                # The system prompt steers Claude to stay focused on the paper
                # and be honest about the limits of the summary in context.
                #
                # Expected stop_reason: "end_turn" — no tools are defined for
                # this call, so Claude always answers directly.
                api_response = _client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    system=_SYSTEM_PROMPT,
                    messages=st.session_state.messages,
                )
                # Extract the text from the first (and only) content block.
                answer = api_response.content[0].text
            st.markdown(answer)

        # Append the assistant's reply to session state so it is included in
        # the history on the next rerun. This is what makes the chat persistent:
        # each turn is saved immediately so subsequent API calls see the full
        # history and can maintain coherent multi-turn conversations.
        st.session_state.messages.append({"role": "assistant", "content": answer})
