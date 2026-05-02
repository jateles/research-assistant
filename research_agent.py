import anthropic
import os
import json
import sys
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def fetch_abstract(pubmed_id: str) -> str:
    response = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={"db": "pubmed", "id": pubmed_id, "rettype": "abstract", "retmode": "text"},
    )
    response.raise_for_status()
    return response.text


def get_pmcid(pmid: str):
    try:
        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi",
            params={"dbfrom": "pubmed", "db": "pmc", "id": pmid, "retmode": "json"},
        )
        data = response.json()
        pmcid = data["linksets"][0]["linksetdbs"][0]["links"][0]
        return str(pmcid)
    except Exception:
        return None


def fetch_full_text(pmid: str) -> str:
    pmcid = get_pmcid(pmid)
    if pmcid:
        try:
            response = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pmc", "id": pmcid, "rettype": "full", "retmode": "xml"},
            )
            response.raise_for_status()
            root = ET.fromstring(response.text)

            sections = []

            abstracts = root.findall(".//abstract")
            if abstracts:
                text = " ".join(" ".join(node.itertext()) for node in abstracts).strip()
                if text:
                    sections.append(f"ABSTRACT\n{text}")

            for label, keywords in [
                ("INTRODUCTION", lambda t: t in ("Introduction", "Background")),
                ("METHODS", lambda t: "Method" in t or "Material" in t),
                ("RESULTS", lambda t: "Result" in t),
                ("DISCUSSION", lambda t: "Discussion" in t),
            ]:
                for sec in root.findall(".//sec"):
                    title_el = sec.find("title")
                    if title_el is not None and title_el.text and keywords(title_el.text.strip()):
                        text = " ".join(sec.itertext()).strip()
                        if text:
                            sections.append(f"{label}\n{text}")
                        break

            figs = root.findall(".//fig")
            if figs:
                fig_lines = []
                for fig in figs:
                    label_el = fig.find("label")
                    caption_el = fig.find("caption")
                    label_text = " ".join(label_el.itertext()).strip() if label_el is not None else ""
                    caption_text = " ".join(caption_el.itertext()).strip() if caption_el is not None else ""
                    entry = ": ".join(x for x in [label_text, caption_text] if x)
                    if entry:
                        fig_lines.append(entry)
                if fig_lines:
                    sections.append("FIGURES\n" + "\n".join(fig_lines))

            output = "\n\n".join(s for s in sections if s)
            return output + "\n\n[Source: Full text via PubMed Central]"
        except Exception:
            return fetch_abstract(pmid) + "\n\n[Source: Abstract only - full text not available]"
    else:
        return fetch_abstract(pmid) + "\n\n[Source: Abstract only - full text not available]"


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
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": f"Can you fetch and summarise the paper for PubMed ID {pubmed_id}?"}
        ],
    )

    if response.stop_reason == "tool_use":
        tool_block = next(b for b in response.content if b.type == "tool_use")
        resolved_id = tool_block.input["pubmed_id"]
        fetched_text = fetch_full_text(resolved_id)

        final_response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            tools=tools,
            messages=[
                {"role": "user", "content": f"Can you fetch and summarise the paper for PubMed ID {pubmed_id}?"},
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": fetched_text,
                        }
                    ],
                },
            ],
        )

        summary = next(b.text for b in final_response.content if b.type == "text")
        return summary, fetched_text

    summary = next((b.text for b in response.content if b.type == "text"), "")
    return summary, ""


def main():
    pubmed_id = sys.argv[1] if len(sys.argv) > 1 else "37651234"
    print("Research agent ready")
    summary, _ = run_agent(pubmed_id)
    print(summary)


if __name__ == "__main__":
    main()
