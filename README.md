# SD Magazine Agent

A small agent that reads San Diego Family Magazine each month and turns it into a digest of free events, family fun days and articles worth reading. It runs on a schedule in GitHub Actions, publishes a browsable guide on GitHub Pages, and delivers the digest as a GitHub issue.

Live guide: https://junihub.github.io/sd-magazine-agent/

## How it works

1. **Find the issue.** `sdfm_agent.py` looks up the current month's issue page on sandiegofamily.com and reads its article list.
2. **Read the calendar.** It visits each day of the magazine's online events calendar, collects the events and marks the free ones. Events listed on five or more days are treated as ongoing and kept once.
3. **Add AI notes.** For each article it asks a language model for one sentence on why a parent might read it. The model is told to work only from the article titles and blurbs.
4. **Write the output.** It saves a Markdown digest to `digests/YYYY-MM.md` and the site data to `issues/YYYY-MM.json`.
5. **Publish.** The workflow commits both files, rebuilds the GitHub Pages site (`index.html` reads the JSON), and creates or updates a GitHub issue with the digest, which GitHub emails to the repository owner.

The workflow in `.github/workflows/monthly.yml` runs on the 1st and 15th of each month and can also be started by hand.

## Cost

Free. In GitHub Actions the AI step uses GitHub Models with the workflow's own token, so there is no API key to manage. On your own computer it falls back to a local Ollama model, and if neither is available the digest is still produced without the AI notes.

## Run it locally

```bash
pip install -r requirements.txt
python sdfm_agent.py            # current month
python sdfm_agent.py 2026 11    # a specific month
```

Optional: if you have the month's magazine as a PDF, save it as `pdfs/YYYY-MM.pdf`. The agent then also reads the printed calendar pages and builds a tab on the site for each article.

## Project structure

- `sdfm_agent.py`: the agent (scraping, calendar parsing, AI notes, output).
- `index.html`: the single-page guide served by GitHub Pages.
- `issues/`: one JSON file per month, plus `index.json` listing the months.
- `digests/`: one Markdown digest per month.
- `.github/workflows/monthly.yml`: the schedule and publishing steps.

## Notes

This is a personal project and is not affiliated with San Diego Family Magazine. It links back to the magazine's own pages for every article and event.
