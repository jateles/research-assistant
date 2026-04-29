import anthropic
import os
import json
import sys
import requests
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


tools = [
    {
        "name": "fetch_abstract",
        "description": "Fetches the abstract of a paper from PubMed given a PubMed ID.",
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


def run_agent(pubmed_id: str) -> str:
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": f"Can you fetch and summarise the abstract for PubMed ID {pubmed_id}?"}
        ],
    )

    if response.stop_reason == "tool_use":
        tool_block = next(b for b in response.content if b.type == "tool_use")
        resolved_id = tool_block.input["pubmed_id"]
        abstract = fetch_abstract(resolved_id)

        final_response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            tools=tools,
            messages=[
                {"role": "user", "content": f"Can you fetch and summarise the abstract for PubMed ID {pubmed_id}?"},
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": abstract,
                        }
                    ],
                },
            ],
        )

        return next(b.text for b in final_response.content if b.type == "text")

    return next((b.text for b in response.content if b.type == "text"), "")


def main():
    pubmed_id = sys.argv[1] if len(sys.argv) > 1 else "37651234"
    print("Research agent ready")
    print(run_agent(pubmed_id))


if __name__ == "__main__":
    main()
