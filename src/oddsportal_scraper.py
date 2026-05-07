"""Scrape Bet365 odds for ATP matches from OddsPortal using Playwright."""

from __future__ import annotations
import asyncio
import re
from playwright.async_api import async_playwright

_BASE = "https://www.oddsportal.com"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


async def _get_match_links(page) -> list[dict]:
    """Return list of {player1, player2, url, time} from /matches/tennis/."""
    await page.goto(f"{_BASE}/matches/tennis/", timeout=30000, wait_until="networkidle")
    await page.wait_for_timeout(2000)

    links = await page.query_selector_all("a")
    matches = []
    for link in links:
        href = await link.get_attribute("href")
        text = (await link.inner_text()).strip()
        if not href or "/tennis/h2h/" not in href:
            continue
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        # Format: [time, player1, score?, –, player2] or similar
        players = [l for l in lines if re.search(r"[A-Z][a-z]", l) and "–" not in l and len(l) > 2]
        if len(players) >= 2:
            matches.append({
                "player1": players[0],
                "player2": players[1],
                "url": f"{_BASE}{href}" if href.startswith("/") else href,
                "text": text,
            })
    return matches


async def _get_bet365_odds(page, match_url: str, p1_raw: str, p2_raw: str) -> dict | None:
    """
    Navigate to a match page and return Bet365 odds correctly ordered for p1/p2.
    Returns {b365w: float, b365l: float} or None if not found.
    """
    await page.goto(match_url, timeout=30000)
    await page.wait_for_timeout(3000)

    body = await page.query_selector("body")
    text = await body.inner_text()
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Detect which player is listed first on the match page (home side)
    p1_last = p1_raw.split()[-1].lower()
    p2_last = p2_raw.split()[-1].lower()
    home_is_p2 = False
    for line in lines[:30]:
        if p2_last in line.lower():
            home_is_p2 = True
            break
        if p1_last in line.lower():
            break

    # Find Bet365 row and extract the two odds (only look AFTER "bet365" line)
    for i, line in enumerate(lines):
        if "bet365" in line.lower():
            # Skip past "CLAIM BONUS" and take the next two float values
            odds_found = []
            for ctx in lines[i + 1: i + 8]:
                try:
                    val = float(ctx)
                    if 1.01 <= val <= 30.0:
                        odds_found.append(val)
                except ValueError:
                    pass
            if len(odds_found) >= 2:
                home_odds, away_odds = odds_found[0], odds_found[1]
                if home_is_p2:
                    return {"b365w": away_odds, "b365l": home_odds}
                return {"b365w": home_odds, "b365l": away_odds}
    return None


def _name_matches(raw: str, target: str) -> bool:
    """Check if a last name from target appears in raw (accent-stripped, case-insensitive)."""
    import unicodedata
    def strip(s):
        return "".join(
            c for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        ).lower()

    target_last = strip(target.split()[-1])
    raw_stripped = strip(raw)
    return target_last in raw_stripped


async def _fetch_odds_async(player_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    """
    Fetch Bet365 odds for a list of (p1_raw, p2_raw) player name pairs.
    Returns {(p1_raw, p2_raw): {b365w, b365l}}.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_extra_http_headers(_HEADERS)

        print("  Fetching match list from OddsPortal...")
        match_links = await _get_match_links(page)
        print(f"  Found {len(match_links)} ATP tennis matches on OddsPortal.")

        results: dict[tuple[str, str], dict] = {}

        for p1_raw, p2_raw in player_pairs:
            # Find matching link
            matched_url = None
            for m in match_links:
                if (
                    _name_matches(m["player1"], p1_raw) and _name_matches(m["player2"], p2_raw)
                ) or (
                    _name_matches(m["player1"], p2_raw) and _name_matches(m["player2"], p1_raw)
                ):
                    matched_url = m["url"]
                    swapped = _name_matches(m["player1"], p2_raw)
                    break

            if not matched_url:
                print(f"  No OddsPortal page found for {p1_raw} vs {p2_raw}")
                continue

            print(f"  Scraping Bet365 odds: {p1_raw} vs {p2_raw}...")
            odds = await _get_bet365_odds(page, matched_url, p1_raw, p2_raw)
            if odds:
                results[(p1_raw, p2_raw)] = odds
                print(f"    → b365w={odds['b365w']}  b365l={odds['b365l']}")
            else:
                print(f"    → Bet365 odds not found")

        await browser.close()
        return results


def fetch_bet365_odds(player_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    """Synchronous wrapper around the async scraper."""
    return asyncio.run(_fetch_odds_async(player_pairs))
