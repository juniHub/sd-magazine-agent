#!/usr/bin/env python3
"""
San Diego Family Magazine monthly agent (100% free).

What it does each run:
  1. Finds the current month's issue page on sandiegofamily.com
  2. Pulls the "Articles in the ... Issue" list (topics worth reading)
  3. Pulls the Events Calendar and keeps entries that mention "free"
  4. Summarizes everything into a Markdown digest (saved to digests/YYYY-MM.md)

Setup:
  pip install requests beautifulsoup4
  Free AI summaries, pick one:
    - GitHub Actions: uses GitHub Models automatically (no key needed, see workflow file)
    - Your own computer: install Ollama from ollama.com, then: ollama pull llama3.2

Run:   python sdfm_agent.py            # current month
       python sdfm_agent.py 2026 11    # a specific month
"""
import datetime as dt
import os
import pathlib
import re
import sys

import requests
from bs4 import BeautifulSoup

BASE = "https://www.sandiegofamily.com"
ISSUES = f"{BASE}/magazines/san-diego-family-digital-issues"
CALENDAR = f"{BASE}/things-to-do/events-calendar"
HEADERS = {"User-Agent": "Mozilla/5.0 (personal monthly digest agent)"}
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
GH_MODELS_URL = "https://models.github.ai/inference/chat/completions"
GH_MODEL = "openai/gpt-4o-mini"


def get_soup(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def find_issue_url(year, month):
    """Issue URLs contain the month/year slug, e.g. san-diego-family-magazine-october-2026."""
    slug = f"san-diego-family-magazine-{dt.date(year, month, 1):%B}-{year}".lower()
    for a in get_soup(ISSUES).find_all("a", href=True):
        if slug in a["href"].lower():
            return requests.compat.urljoin(BASE, a["href"])
    return None


def get_articles(issue_url):
    """Grab the bullet list under 'Articles in the <Month> Issue'."""
    soup = get_soup(issue_url)
    heading = soup.find(lambda t: t.name in ("h2", "h3", "h4") and "Articles in the" in t.get_text())
    if not heading:
        return []
    ul = heading.find_next("ul")
    return [li.get_text(" ", strip=True) for li in ul.find_all("li")] if ul else []


def get_free_events(year, month):
    """Keep calendar lines that mention 'free'. Adjust selectors if the site layout changes."""
    text = get_soup(CALENDAR).get_text("\n", strip=True)
    lines = [l for l in text.split("\n") if len(l) > 15]
    free = [l for l in lines if re.search(r"\bfree\b", l, re.I)]
    seen, out = set(), []
    for l in free:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return out[:40]


def summarize(prompt, fallback):
    """Free AI summary. Tries GitHub Models (inside GitHub Actions), then local Ollama,
    and finally falls back to the plain text so the script never breaks."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        try:
            r = requests.post(
                GH_MODELS_URL,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"model": GH_MODEL, "messages": [{"role": "user", "content": prompt}]},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"GitHub Models failed ({e}); trying Ollama...", file=sys.stderr)
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["response"].strip()
    except Exception:
        return fallback


def main():
    today = dt.date.today()
    year = int(sys.argv[1]) if len(sys.argv) > 1 else today.year
    month = int(sys.argv[2]) if len(sys.argv) > 2 else today.month
    label = f"{dt.date(year, month, 1):%B %Y}"

    issue_url = find_issue_url(year, month)
    articles = get_articles(issue_url) if issue_url else []
    events = get_free_events(year, month)

    article_md = summarize(
        "You are a helpful parenting-magazine curator. For each article below, write one "
        "friendly sentence on why a San Diego parent might want to read it, and tag it "
        "(Parenting / Activities / Crafts / Community). Keep the page numbers.\n\n"
        + "\n".join(articles),
        "\n".join(f"- {a}" for a in articles) or "_No article list found._",
    ) if articles else "_No article list found._"

    event_md = summarize(
        f"From these San Diego event listings, pick the best FREE family events for {label}. "
        "Group them by date, one line each with the name and where. Do not invent details.\n\n"
        + "\n".join(events),
        "\n".join(f"- {e}" for e in events) or "_No free events detected._",
    ) if events else "_No free events detected._"

    digest = (
        f"# San Diego Family Magazine, {label}\n\n"
        f"Issue: {issue_url or 'not found yet'}\n\n"
        f"## Worth reading\n{article_md}\n\n"
        f"## Free events\n{event_md}\n\n"
        f"Full calendar: {CALENDAR}\n"
    )

    out = pathlib.Path("digests")
    out.mkdir(exist_ok=True)
    path = out / f"{year}-{month:02d}.md"
    path.write_text(digest, encoding="utf-8")
    print(digest)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
