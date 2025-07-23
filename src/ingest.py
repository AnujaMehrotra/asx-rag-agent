import requests
from bs4 import BeautifulSoup
import os

BANKS = {
    "cba": "https://en.wikipedia.org/wiki/Commonwealth_Bank",
    "wbc": "https://en.wikipedia.org/wiki/Westpac",
    "nab": "https://en.wikipedia.org/wiki/National_Australia_Bank",
    "anz": "https://en.wikipedia.org/wiki/Australia_and_New_Zealand_Banking_Group"
}

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_wikipedia_summary(url: str) -> str:
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    
    paragraphs = soup.select("p")
    summary = ""

    for p in paragraphs:
        text = p.get_text(strip=True)
        if text and not text.lower().startswith("coordinates"):
            summary += text + "\n"
        if len(summary.split()) > 200:  # limit to ~200 words
            break

    return summary.strip()

if __name__ == "__main__":
    for code, url in BANKS.items():
        print(f"Fetching {code.upper()} Wikipedia page...")
        summary = fetch_wikipedia_summary(url)
        filename = os.path.join(OUTPUT_DIR, f"{code}_wikipedia.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Saved summary to {filename}")
