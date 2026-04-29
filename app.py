import unicodedata
import streamlit as st
from fpdf import FPDF
from research_agent import run_agent

st.title("Research Assistant")

pmid = st.text_input("PubMed ID")

if st.button("Fetch & Summarise"):
    if not pmid.strip():
        st.warning("Please enter a PubMed ID.")
    else:
        with st.spinner("Fetching and summarising..."):
            summary = run_agent(pmid.strip())

        st.markdown(summary)

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="Download as Markdown",
                data=summary,
                file_name=f"{pmid}.md",
                mime="text/markdown",
            )

        with col2:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("helvetica", size=12)
            clean_summary = unicodedata.normalize("NFKD", summary).encode("latin-1", "ignore").decode("latin-1")
            pdf.multi_cell(0, 10, clean_summary)
            pdf_bytes = pdf.output()

            st.download_button(
                label="Download as PDF",
                data=bytes(pdf_bytes),
                file_name=f"{pmid}.pdf",
                mime="application/pdf",
            )
