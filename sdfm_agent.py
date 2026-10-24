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
import calendar
import datetime as dt
import json
import os
import pathlib
import re
import sys
import time
from urllib.parse import quote_plus, urljoin

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


# Verified section pages on the site, used when an article has no link of its own.
SECTION_LINKS = [
    (("calendar",), f"{BASE}/things-to-do/events-calendar"),
    (("pumpkin", "fall fun", "seasonal"), f"{BASE}/things-to-do/seasonal-happenings"),
    (("craft",), f"{BASE}/for-the-kids/crafts-for-kids"),
    (("homework", "friend", "buddies", "besties"), f"{BASE}/parenting/big-kid"),
    (("afterschool",), f"{BASE}/resources/afterschool-activities"),
]


def best_link(title, blurb):
    """Pick the best link for an article that has no direct link."""
    text = f"{title} {blurb}".lower()
    for keywords, url in SECTION_LINKS:
        if any(k in text for k in keywords):
            return url, "section"
    q = quote_plus(f'site:sandiegofamily.com {title}')
    return f"https://www.google.com/search?q={q}", "search"


def get_articles(issue_url):
    """Return a list of dicts: page, title, blurb, url, kind, from the 'Articles in the ... Issue' list."""
    soup = get_soup(issue_url)
    heading = soup.find(lambda t: t.name in ("h2", "h3", "h4") and "Articles in the" in t.get_text())
    ul = heading.find_next("ul") if heading else None
    items = []
    for li in (ul.find_all("li") if ul else []):
        strong = li.find(["strong", "b"])
        head = strong.get_text(" ", strip=True) if strong else li.get_text(" ", strip=True)
        full = li.get_text(" ", strip=True)
        blurb = full.replace(head, "", 1).strip(" \u2013-:") if strong else ""
        m = re.match(r"Page\s*(\d+)\s*:\s*(.*)", head)
        page = m.group(1) if m else ""
        title = (m.group(2) if m else head).strip(" \u2013-:")
        a = li.find("a", href=True)
        if a:
            url, kind = urljoin(BASE, a["href"]), "direct"
        else:
            url, kind = best_link(title, blurb)
        items.append({"page": page, "title": title, "blurb": blurb, "url": url, "kind": kind})
    return items


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


def summarize(prompt, fallback, max_tokens=None):
    """Free AI summary. Tries GitHub Models (inside GitHub Actions), then local Ollama,
    and finally falls back to the plain text so the script never breaks."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        try:
            r = requests.post(
                GH_MODELS_URL,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"model": GH_MODEL, "messages": [{"role": "user", "content": prompt}],
                      **({"max_tokens": max_tokens} if max_tokens else {})},
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


def format_articles(articles, issue_url):
    """Markdown list with a link per article plus an optional one-line AI 'why read it'."""
    if not articles:
        return "_No article list found._"
    numbered = "\n".join(f"{i}. {x['title']} - {x['blurb']}" for i, x in enumerate(articles, 1))
    reply = summarize(
        "You curate a parenting magazine for San Diego parents. For EACH numbered article, "
        "write ONE friendly sentence (max 20 words) on why a parent might want to read it. "
        "Reply with the same numbers, one per line, nothing else. Do not invent facts.\n\n"
        + numbered,
        "",
    )
    why = {}
    for line in reply.splitlines():
        m = re.match(r"^\s*(\d+)[.)]\s*(.+)$", line)
        if m:
            why[int(m.group(1))] = m.group(2).strip()
    for i, x in enumerate(articles, 1):
        x["why"] = why.get(i, "")
    labels = {"direct": "", "section": " (related section)", "search": " (search)"}
    out = []
    for i, x in enumerate(articles, 1):
        page = f" (p. {x['page']}, [read in magazine]({issue_url}))" if x["page"] and issue_url else ""
        line = f"- **[{x['title']}]({x['url']})**{labels[x['kind']]}{page}"
        if x["blurb"]:
            line += f" - {x['blurb']}"
        if why.get(i):
            line += f"\n  - Why read: {why[i]}"
        out.append(line)
    return "\n".join(out)


def day_text(year, month, day):
    """Text of the magazine site's page for ONE day (it lists that day's events with FREE categories)."""
    url = f"{BASE}/things-to-do/events-calendar/day/{month:02d}-{day:02d}-{year}"
    try:
        text = get_soup(url).get_text("\n", strip=True)
    except Exception:
        return ""
    i = text.find("Events from")
    return text[i:i + 2500] if i >= 0 else ""


def extract_events(year, month):
    """Read every day page of the month, then ask the AI to structure each day's events.
    Dates come from the page address (not guessed). Returns {day: [event, ...]}."""
    days = calendar.monthrange(year, month)[1]
    blocks = []
    for d in range(1, days + 1):
        t = day_text(year, month, d)
        if "Categories" in t:
            blocks.append((d, t))
    events = {}
    for i in range(0, len(blocks), 6):  # 6 days per AI call keeps prompts small
        group = blocks[i:i + 6]
        reply = summarize(
            "Each block below is one day of a San Diego family events calendar. Extract the events. "
            "Return ONLY JSON shaped {\"<day number>\": [{\"name\":..,\"icon\":\"one emoji\","
            "\"type\":\"free|paid|show|outdoor\",\"time\":..,\"loc\":..,\"desc\":\"one sentence\","
            "\"url\":\"\",\"free\":true|false}]}. Use ONLY facts in the text. "
            "free=true if the Categories include FREE or the description says free. "
            "Set url only if a web address appears in the text (add https://), else empty string.\n\n"
            + "\n\n".join(f"=== DAY {d} ===\n{t}" for d, t in group),
            "", max_tokens=4000,
        )
        try:
            raw = json.loads(reply[reply.index("{"): reply.rindex("}") + 1])
            for k, v in raw.items():
                if isinstance(v, list):
                    events.setdefault(str(int(k)), []).extend(v)
        except Exception:
            print(f"AI returned nothing usable for days {[d for d, _ in group]}", file=sys.stderr)
        time.sleep(2)
    return events


def write_site_data(year, month, issue_url, articles, events):
    """Write issues/YYYY-MM.json (what index.html reads) and refresh issues/index.json."""
    d = pathlib.Path("issues")
    d.mkdir(exist_ok=True)
    key = f"{year}-{month:02d}"
    (d / f"{key}.json").write_text(json.dumps({
        "month": key, "label": f"{dt.date(year, month, 1):%B %Y}", "subtitle": "",
        "issue_url": issue_url or "", "articles": articles, "events": events, "camps": [],
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    idx = d / "index.json"
    months = set(json.loads(idx.read_text())) if idx.exists() else set()
    months.add(key)
    idx.write_text(json.dumps(sorted(months)))


def main():
    today = dt.date.today()
    year = int(sys.argv[1]) if len(sys.argv) > 1 else today.year
    month = int(sys.argv[2]) if len(sys.argv) > 2 else today.month
    label = f"{dt.date(year, month, 1):%B %Y}"

    issue_url = find_issue_url(year, month)
    articles = get_articles(issue_url) if issue_url else []
    events = get_free_events(year, month)

    article_md = format_articles(articles, issue_url)

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

    write_site_data(year, month, issue_url, articles, extract_events(year, month))

    out = pathlib.Path("digests")
    out.mkdir(exist_ok=True)
    path = out / f"{year}-{month:02d}.md"
    path.write_text(digest, encoding="utf-8")
    print(digest)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
