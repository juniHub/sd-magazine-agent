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
        page = f" (p. {x['page']}, [read in magazine]({x.get('page_url') or issue_url}))" if x["page"] and issue_url else ""
        line = f"- **[{x['title']}]({x['url']})**{labels[x['kind']]}{page}"
        if x["blurb"]:
            line += f" - {x['blurb']}"
        if why.get(i):
            line += f"\n  - Why read: {why[i]}"
        out.append(line)
    return "\n".join(out)


TIME_RE = re.compile(r"\d{1,2}(?::\d{2})?\s*(?:am|pm)(?:\s*[\u2013-]\s*\d{1,2}(?::\d{2})?\s*(?:am|pm))?", re.I)
WEB_RE = re.compile(r"(?:https?://|www\.)[^\s,;)]+", re.I)
ICONS = [("free", "\U0001F193"), ("halloween", "\U0001F383"), ("concert", "\U0001F3B5"), ("museum", "\U0001F3DB\uFE0F"),
         ("amusement", "\U0001F3A2"), ("skating", "\u26F8\uFE0F"), ("muertos", "\U0001F480"), ("tea", "\u2615")]


def parse_day(year, month, day):
    """Read one day page on the magazine's calendar. Every event is a list item with a title link,
    location, category links (incl. FREE) and a description, so no AI is needed."""
    url = f"{BASE}/things-to-do/events-calendar/day/{month:02d}-{day:02d}-{year}"
    try:
        soup = get_soup(url)
    except Exception:
        return []
    out, seen = [], set()
    for a in soup.find_all("a", href=re.compile(r"/events-calendar/event/")):
        title, li = a.get_text(" ", strip=True), a.find_parent("li")
        if not title or not li or title in seen:
            continue
        seen.add(title)
        cats = [c.get_text(strip=True) for c in li.find_all("a", href=re.compile(r"/events-calendar/category/"))]
        loc = li.find("a", href=re.compile(r"/events-calendar/location/"))
        lines = li.get_text("\n", strip=True).split("\n")
        desc = lines[-1]
        if cats and cats[-1] in lines:
            desc = " ".join(lines[len(lines) - 1 - lines[::-1].index(cats[-1]) + 1:]) or desc
        for prefix in ("free " + title, title):
            if desc.lower().startswith(prefix.lower()):
                desc = desc[len(prefix):]
                break
        desc = desc.lstrip(" .:").strip()
        free = any(c.upper() == "FREE" for c in cats) or desc.upper().startswith("FREE")
        w = WEB_RE.search(desc)
        site = w.group(0).rstrip(".") if w else ""
        if site and not site.lower().startswith("http"):
            site = "https://" + site
        t = TIME_RE.search(desc)
        joined = " ".join(cats).lower()
        out.append({
            "name": title, "free": free, "type": "free" if free else "other",
            "icon": next((i for k, i in ICONS if k in joined or (k == "free" and free)), "\U0001F4C5"),
            "time": t.group(0) if t else "", "loc": loc.get_text(" ", strip=True) if loc else "",
            "desc": desc[:300], "url": site or urljoin(BASE, a["href"]),
        })
    return out


def extract_events(year, month):
    """Returns ({day: [events]}, [ongoing events]). An event listed on 5+ days is 'ongoing'
    (e.g. 'Through Oct. 31') and is kept once instead of being repeated every day."""
    seen = {}
    for d in range(1, calendar.monthrange(year, month)[1] + 1):
        for e in parse_day(year, month, d):
            seen.setdefault(e["name"], []).append((d, e))
        time.sleep(0.3)
    events, ongoing = {}, []
    for items in seen.values():
        if len(items) >= 5:
            e = dict(items[0][1])
            e["days"] = [d for d, _ in items]
            ongoing.append(e)
        else:
            for d, e in items:
                events.setdefault(str(d), []).append(e)
    return events, ongoing


def free_markdown(events, ongoing, year, month):
    mon = f"{dt.date(year, month, 1):%b}"
    link = lambda e: f"[{e['name']}]({e['url']})"
    lines = [f"- {link(e)} (ongoing) - {e['loc']}" for e in ongoing if e["free"]]
    for d in sorted(events, key=int):
        lines += [f"- {mon} {d}: {link(e)} - {e['loc']}" for e in events[d] if e["free"]]
    return "\n".join(lines) or "_No free events detected._"


SKIP_TITLES = ("calendar", "afterschool")
ICON_RULES = [("pumpkin", "\U0001F383"), ("halloween", "\U0001F383"), ("muertos", "\U0001F480"), ("craft", "\U0001F3A8"),
              ("homework", "\U0001F4DA"), ("friend", "\U0001F46B"), ("4-h", "\U0001F331"), ("tradition", "\U0001F342"),
              ("camp", "\U0001F3D5\uFE0F"), ("holiday", "\U0001F384"), ("summer", "\u2600\uFE0F")]


def pdf_sections(year, month, articles):
    """If pdfs/YYYY-MM.pdf exists, turn each magazine article into a site tab using its page range.
    The AI only reorganizes the article's own text into JSON; it is told not to add facts."""
    path = pathlib.Path("pdfs") / f"{year}-{month:02d}.pdf"
    if not path.exists():
        print(f"No {path}; skipping article tabs.", file=sys.stderr)
        return []
    try:
        from pypdf import PdfReader
        pages = [(p.extract_text() or "") for p in PdfReader(str(path)).pages]
    except Exception as e:
        print(f"Could not read {path}: {e}", file=sys.stderr)
        return []
    starts = sorted({int(a["page"]) for a in articles if a.get("page")})
    out = []
    for a in articles:
        title = a["title"]
        if not a.get("page") or any(k in title.lower() for k in SKIP_TITLES):
            continue
        start = int(a["page"])
        later = [p for p in starts if p > start]
        end = min((later[0] - 1) if later else start + 1, len(pages))
        text = "\n".join(pages[i - 1] for i in range(start, end + 1))[:9000]
        reply = summarize(
            f"Magazine article '{title}' ({a.get('blurb', '')}). Turn its text into JSON only: "
            '{"intro":"1-2 sentences","items":[{"name":"","meta":"dates/price/FREE","desc":"","url":"","loc":""}],'
            '"tips":["short tips"]}. Use ONLY facts in the text. Ignore ads, page headers and unrelated listings. '
            "Use empty strings for missing fields. Set url only if a web address appears in the text (add https://).\n\n"
            + text, "", max_tokens=1800)
        try:
            j = json.loads(reply[reply.index("{"): reply.rindex("}") + 1])
        except Exception:
            print(f"AI gave no usable JSON for '{title}'", file=sys.stderr)
            continue
        icon = next((i for k, i in ICON_RULES if k in title.lower()), "\U0001F4F0")
        out.append({"title": title, "icon": icon, "intro": j.get("intro", ""),
                    "items": [i for i in j.get("items", []) if i.get("name")],
                    "tips": j.get("tips", []), "note": f"From page {start} of the magazine."})
        time.sleep(2)
    return out


def issuu_url(year, month):
    """The magazine's Issuu flipbook address follows a pattern. Only a 404 means it isn't there."""
    url = f"https://issuu.com/sandiegofamilymagazine/docs/san_diego_family_{dt.date(year, month, 1):%B}_{year}".lower()
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        return url if r.status_code != 404 else None
    except Exception:
        return None


PAGE_OFFSET = 0  # if links land one page off, change this to 1 or -1


def reader_url(read_url, page):
    """Issuu full-screen reader link that opens at a given page."""
    slug = read_url.rstrip("/").rsplit("/", 1)[-1]
    return f"https://issuu.com/rd4?p={int(page) + PAGE_OFFSET}&d={slug}&u=sandiegofamilymagazine"


def issuu_overview(url):
    """The publisher's own public description of the issue (page meta text), or ''."""
    try:
        soup = get_soup(url)
        for attrs in ({"property": "og:description"}, {"name": "description"}):
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content") and len(tag["content"]) > 60:
                return tag["content"].strip()
    except Exception:
        pass
    return ""


DAY_HDR = re.compile(r"^(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*\|\s*([\d ]+?)\s*$", re.I)
URL_RE = re.compile(r"(?:https?://\S+|www\.\S+|\b[\w.-]+\.(?:com|org|net|gov|edu|biz|church)\b\S*)", re.I)
TIME_RANGE = re.compile(r"\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s*[\u2013-]\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)", re.I)
WD = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
WD_RE = r"(?:mon|tues?|wed|thu(?:rs?)?|fri|sat|sun)"


def _wd(tok):
    return WD[tok.lower()[:3]]


def weekdays_in(text):
    """Weekdays an event runs on, from phrases like 'Weekends', 'Fridays & Saturdays', 'Fri.-Sun.', 'Sat. & Sun.'."""
    out = set()
    if re.search(r"\bweekends?\b", text, re.I):
        out |= {5, 6}
    for m in re.finditer(rf"\b({WD_RE})\.?\s*[\u2013-]\s*({WD_RE})\b", text, re.I):
        a, b = _wd(m.group(1)), _wd(m.group(2))
        out |= {(a + i) % 7 for i in range(((b - a) % 7) + 1)}
    for m in re.finditer(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)s\b", text, re.I):
        out.add(_wd(m.group(1)))
    for m in re.finditer(rf"\b((?:{WD_RE})\.(?:\s*(?:,|&|and)\s*(?:{WD_RE})\.)+)", text, re.I):
        out |= {_wd(t) for t in re.findall(WD_RE, m.group(1), re.I)}
    return out


def explicit_days(text, month):
    """Day numbers written like 'Oct. 17 & 31', 'October 23-24' or 'Oct. 3, 9, 10' (ignores times like '10-11:30 am')."""
    name = dt.date(2000, month, 1).strftime("%B")
    days = set()
    for m in re.finditer(rf"\b(?:{name[:3]}\.|{name})\s+(?=\d)", text):
        if re.search(r"(?:through|until|thru)\s+$", text[:m.start()], re.I):
            continue  # an end date, not an event date
        pos = m.end()
        while True:
            it = re.match(r"(\d{1,2})(?:\s*[\u2013-]\s*(\d{1,2}))?", text[pos:])
            if not it:
                break
            after = text[pos + it.end(): pos + it.end() + 12]
            if re.match(r"\s*(?::|am\b|pm\b|a\.m|p\.m)", after):
                break
            a, b = int(it.group(1)), int(it.group(2) or it.group(1))
            days |= set(range(a, b + 1)) if b >= a else {a}
            sep = re.match(r"\s*(?:,|&|and)\s*(?=\d)", text[pos + it.end():])
            if not sep:
                break
            pos += it.end() + sep.end()
    return {d for d in days if 1 <= d <= 31}


def expand_days(text, start, year, month):
    """Every day an entry runs, from its listed day plus phrases like 'Open daily', 'through Oct. 11', 'Today & tomorrow'."""
    last = calendar.monthrange(year, month)[1]
    days = {d for d in explicit_days(text, month) if d <= last}
    if days and start not in days and not re.search(r"\btoday\b", text, re.I):
        return sorted(days)  # featured box with its own dates (e.g. 'Oct. 17, 5-8 pm') sitting under another day
    days.add(start)
    if re.search(r"\btomorrow\b", text, re.I) and start < last:
        days.add(start + 1)
    if re.search(r"select (?:dates|days|saturdays|sundays|fridays)|quarterly|rotating basis", text, re.I):
        return sorted(days)
    closed = set()
    cm = re.search(rf"closed\s+({WD_RE})\.?(?:\s*[\u2013-]\s*({WD_RE})\b)?", text, re.I)
    if cm:
        a = _wd(cm.group(1)); b = _wd(cm.group(2)) if cm.group(2) else a
        closed = {(a + i) % 7 for i in range(((b - a) % 7) + 1)}
        text = text.replace(cm.group(0), " ")
    wds = weekdays_in(text)
    thr = re.search(rf"through\s+(?:{dt.date(2000, month, 1):%b}\.|{dt.date(2000, month, 1):%B})\s+(\d{{1,2}})", text, re.I)
    ongoing = thr or re.search(r"through\s+(?:[A-Z][a-z]{2,8}\.?)\s+\d", text) or re.search(r"\b(open daily|daily)\b", text, re.I) or wds
    if ongoing and not (wds and not thr and not re.search(r"through|daily|weekends?", text, re.I) and False):
        end = min(int(thr.group(1)), last) if thr else last
        for d in range(start, end + 1):
            if (not wds or calendar.weekday(year, month, d) in wds) and calendar.weekday(year, month, d) not in closed:
                days.add(d)
    return sorted(d for d in days if calendar.weekday(year, month, d) not in closed or d == start)


def _event_from_lines(buf, day, year, month):
    text = ""
    for l in buf:
        text += l if text.endswith(("/", "-", "_")) else (" " + l if text else l)
    text = re.sub(r"\s+", " ", text).strip()
    free = bool(re.match(r"^[^.]{0,50}?\bFREE\b", text))
    body = re.sub(r"^FREE\s+", "", text)
    first = re.sub(r"^FREE\s+", "", buf[0].strip())
    m = re.match(r"^(.{3,90}?\.[\u201d\"]?)\s", body)
    minor = {"at", "of", "the", "and", "in", "for", "a", "to", "on", "with", "de", "los", "la", "&", "by", "from"}
    heading_ok = (len(first.split()) <= 7 and "." not in first and first[:1].isupper() and not first.isupper()
                  and all(w[:1].isupper() or not w[:1].isalpha() or w.lower() in minor for w in first.split())
                  and re.search(r"\b(?:am|pm|oct\.?|october|weekends?|daily|through|today)\b", text, re.I))
    # a short heading line followed by a long sentence (feature box) vs. a normal "Title. Description" entry
    if heading_ok and (not m or len(m.group(1)) > len(first) + 45):
        title = first
    elif m:
        title = m.group(1)
    else:
        return None
    desc = body[len(title):].strip() if body.startswith(title) else body
    title = title.strip(" .\u201c\u201d\"")
    if len(title) < 3:
        return None
    urls = URL_RE.findall(text)
    url = urls[-1].rstrip(".,;)") if urls else ""
    if url and not url.lower().startswith("http"):
        url = "https://" + url
    t = TIME_RANGE.search(desc) or TIME_RE.search(desc)
    days = expand_days(text, day, year, month)
    return {"name": title, "free": free, "type": "free" if free else "other",
            "icon": "\U0001F193" if free else "\U0001F4C5", "time": t.group(0) if t else "", "loc": "",
            "desc": desc[:300], "url": url, "days": days}


def parse_pdf_calendar(path, year, month, start_page=None):
    """Read the month's calendar pages of the magazine PDF. Returns ({day: [events]}, [ongoing events])."""
    from pypdf import PdfReader
    pages = [(p.extract_text() or "") for p in PdfReader(str(path)).pages]
    first = (start_page or 1) - 1
    cur, entries = 1, []
    for i in range(first, len(pages)):
        lines = [l.strip() for l in pages[i].split("\n")]
        if not any(DAY_HDR.match(l) for l in lines):
            continue
        buf, wrapped = [], False
        for l in lines:
            if (not l or "sandiegofamily.com" in l.lower() or re.fullmatch(r"(C|M|Y|CM|MY|CY|CMY|K)", l)
                    or ".pdf" in l or re.match(r"^Calendar\b", l)):
                continue
            h = DAY_HDR.match(l)
            if h:
                buf, cur = [], int(h.group(1).replace(" ", ""))
                continue
            if l.startswith("FREE ") and buf:
                entries.append((cur, buf)); buf = []
            buf.append(l)
            if wrapped:
                entries.append((cur, buf)); buf, wrapped = [], False
            elif URL_RE.search(l):
                if l.endswith(("/", "-", "_")):
                    wrapped = True
                else:
                    entries.append((cur, buf)); buf = []
        if any("Find more events at" in l for l in lines):
            break
    seen, events, ongoing = set(), {}, []
    for day, buf in entries:
        e = _event_from_lines(buf, day, year, month)
        if not e or (e["name"].lower(), tuple(e["days"])) in seen:
            continue
        seen.add((e["name"].lower(), tuple(e["days"])))
        days = e.pop("days")
        if len(days) >= 5:
            e["days"] = days
            ongoing.append(e)
        else:
            for d in days:
                events.setdefault(str(d), []).append(dict(e))
    return events, ongoing


def _key(name):
    return re.sub(r"[^a-z0-9]", "", re.sub(r"^free\s+", "", name.lower()))


def merge_events(ev_a, og_a, ev_b, og_b):
    """Union of two event sets. Same-named events on a day are kept once; ongoing events merge their days."""
    ev = {d: list(v) for d, v in ev_a.items()}
    for d, lst in ev_b.items():
        have = {_key(e["name"]) for e in ev.get(d, [])}
        for e in lst:
            if _key(e["name"]) not in have:
                ev.setdefault(d, []).append(e)
    og = {_key(e["name"]): dict(e) for e in og_a}
    for e in og_b:
        k = _key(e["name"])
        if k in og:
            og[k]["days"] = sorted(set(og[k].get("days", [])) | set(e.get("days", [])))
        else:
            og[k] = dict(e)
    return ev, list(og.values())


def write_site_data(year, month, issue_url, articles, events, ongoing, sections=None, overview=""):
    """Write issues/YYYY-MM.json (what index.html reads) and refresh issues/index.json."""
    d = pathlib.Path("issues")
    d.mkdir(exist_ok=True)
    key = f"{year}-{month:02d}"
    f = d / f"{key}.json"
    old = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    events, ongoing = merge_events(events, ongoing, old.get("events", {}), old.get("ongoing", []))
    f.write_text(json.dumps({  # hand-made parts (subtitle, camps, sections) survive each re-run
        "month": key, "label": f"{dt.date(year, month, 1):%B %Y}", "subtitle": old.get("subtitle", ""),
        "issue_url": issue_url or "", "articles": articles, "events": events, "ongoing": ongoing,
        "overview": overview or old.get("overview", ""), "camps": old.get("camps", []), "sections": old.get("sections") or sections or [],
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    idx = d / "index.json"
    months = set(json.loads(idx.read_text())) if idx.exists() else set()
    months.add(key)
    idx.write_text(json.dumps(sorted(months)))
    return events, ongoing  # merged with the saved month, so the digest covers the whole month


def main():
    today = dt.date.today()
    year = int(sys.argv[1]) if len(sys.argv) > 1 else today.year
    month = int(sys.argv[2]) if len(sys.argv) > 2 else today.month
    label = f"{dt.date(year, month, 1):%B %Y}"

    issue_url = find_issue_url(year, month)
    articles = get_articles(issue_url) if issue_url else []
    events, ongoing = extract_events(year, month)
    pdf = pathlib.Path("pdfs") / f"{year}-{month:02d}.pdf"
    if pdf.exists():
        try:
            cal_page = next((int(a["page"]) for a in articles if a.get("page") and "calendar" in a["title"].lower()), None)
            pe, po = parse_pdf_calendar(pdf, year, month, cal_page)
            events, ongoing = merge_events(pe, po, events, ongoing)
        except Exception as e:
            print(f"Could not read calendar from {pdf}: {e}", file=sys.stderr)

    read_url = issuu_url(year, month) or issue_url
    overview = issuu_overview(read_url) if read_url and "issuu.com" in read_url else ""
    if read_url and "issuu.com" in read_url:
        for x in articles:
            if x.get("page"):
                x["page_url"] = reader_url(read_url, x["page"])
    article_md = format_articles(articles, read_url)
    events, ongoing = write_site_data(year, month, read_url, articles, events, ongoing,
                                      pdf_sections(year, month, articles), overview)
    event_md = free_markdown(events, ongoing, year, month)

    digest = (
        f"# San Diego Family Magazine, {label}\n\n"
        f"Issue: {issue_url or 'not found yet'}\n\n"
        f"## Worth reading\n{article_md}\n\n"
        f"## Free events (whole month)\n{event_md}\n\n"
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
