"""Fetch ATP rankings, latest match scores, and upcoming matches.

Rankings/scores: atptour.com (requests + BeautifulSoup / __NEXT_DATA__).
Upcoming matches: Sofascore unofficial API (JSON, no JS rendering required).
Raises RuntimeError with a user-friendly message if parsing fails.
"""

from __future__ import annotations

import datetime
import json
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.atptour.com/",
}

_SOFASCORE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.sofascore.com",
}

ATP_BASE = "https://www.atptour.com"
_SOFASCORE_BASE = "https://www.sofascore.com/api/v1"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get(url: str, **kwargs) -> requests.Response:
    resp = requests.get(url, headers=_HEADERS, timeout=20, **kwargs)
    resp.raise_for_status()
    return resp


def _extract_next_data(soup: BeautifulSoup) -> dict:
    """Return parsed __NEXT_DATA__ JSON if the page is a Next.js app."""
    tag = soup.find("script", id="__NEXT_DATA__")
    if tag and tag.string:
        try:
            return json.loads(tag.string)
        except json.JSONDecodeError:
            pass
    return {}


def _deep_search(obj, *keys):
    """Walk a nested dict/list trying multiple key names; return first hit."""
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                val = obj[k]
                if val:
                    return val
        for v in obj.values():
            result = _deep_search(v, *keys)
            if result:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _deep_search(item, *keys)
            if result:
                return result
    return None


def _parse_rankings_from_table(soup: BeautifulSoup, top_n: int) -> list[dict]:
    """HTML-table fallback for the rankings page."""
    table = (
        soup.find("table", class_=re.compile(r"mega-table|ranking", re.I))
        or soup.find("table")
    )
    if not table:
        return []

    rows: list[dict] = []
    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr")[:top_n]:
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        try:
            rank_text = tds[0].get_text(strip=True)
            rank_match = re.search(r"\d+", rank_text)
            if not rank_match:
                continue
            rank = int(rank_match.group())

            # Player name — prefer the first anchor link with substantial text
            player = ""
            for td in tds:
                link = td.find("a")
                if link:
                    name = link.get_text(strip=True)
                    if len(name) > 3:
                        player = name
                        break
            if not player:
                player = tds[min(2, len(tds) - 1)].get_text(strip=True)

            country = tds[1].get_text(strip=True) if len(tds) > 1 else ""

            # Points — look for the first large integer in remaining cells
            points = 0
            for td in tds[2:7]:
                val = td.get_text(strip=True).replace(",", "").replace(".", "")
                if val.isdigit() and int(val) > 50:
                    points = int(val)
                    break

            if player:
                rows.append({"rank": rank, "player": player, "country": country, "points": points})
        except (ValueError, IndexError):
            continue
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_atp_rankings(top_n: int = 100) -> pd.DataFrame:
    """
    Fetch current ATP singles rankings from Sofascore (ranking ID 5).

    Returns
    -------
    DataFrame with columns: rank (int) | player (str) | country (str) | points (int)

    Raises
    ------
    RuntimeError  if the API response cannot be parsed.
    requests.HTTPError / requests.ConnectionError  on network failure.
    """
    url = f"{_SOFASCORE_BASE}/rankings/5"
    resp = requests.get(url, headers=_SOFASCORE_HEADERS, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    ranking_rows = data.get("rankingRows", [])
    if not ranking_rows:
        raise RuntimeError("Sofascore ATP rankings response contained no rows.")

    rows: list[dict] = []
    for item in ranking_rows[:top_n]:
        team = item.get("team", {})
        country_info = team.get("country", {})
        rows.append({
            "rank": item.get("position"),
            "player": team.get("name", "").strip(),
            "country": country_info.get("alpha2", "") if isinstance(country_info, dict) else "",
            "points": item.get("points", 0),
        })

    df = pd.DataFrame(rows)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0).astype(int)
    return df.dropna(subset=["rank"]).sort_values("rank").reset_index(drop=True)


def fetch_atp_scores() -> pd.DataFrame:
    """
    Fetch the latest ATP match results from atptour.com/en/scores/results.

    Returns
    -------
    DataFrame with columns: tournament | surface | round | player1 | player2 | score

    Raises
    ------
    RuntimeError  if the page cannot be parsed.
    """
    url = f"{ATP_BASE}/en/scores/results"
    resp = _get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    matches: list[dict] = []

    # ── Try Next.js embedded data ─────────────────────────────────────────────
    next_data = _extract_next_data(soup)
    if next_data:
        score_list = _deep_search(
            next_data,
            "scoresData", "matches", "results", "score", "scores",
        )
        if isinstance(score_list, list) and score_list:
            for m in score_list:
                if not isinstance(m, dict):
                    continue
                matches.append({
                    "tournament": (
                        m.get("tournament") or m.get("tournamentName")
                        or m.get("eventName") or ""
                    ),
                    "surface": m.get("surface") or m.get("courtSurface") or "",
                    "round": m.get("round") or m.get("roundName") or "",
                    "player1": (
                        m.get("winner") or m.get("player1")
                        or m.get("winnerName") or ""
                    ),
                    "player2": (
                        m.get("loser") or m.get("player2")
                        or m.get("loserName") or ""
                    ),
                    "score": m.get("score") or m.get("result") or "",
                })

    # ── Fallback: pandas read_html ────────────────────────────────────────────
    if not matches:
        try:
            tables = pd.read_html(resp.text)
            for t in tables:
                if len(t.columns) >= 3 and len(t) >= 3:
                    return t.head(50)
        except Exception:
            pass

    if not matches:
        raise RuntimeError(
            "Could not parse ATP Tour scores — the page likely requires JavaScript. "
            "Visit atptour.com/en/scores/results directly for the latest results."
        )

    return pd.DataFrame(matches)


def fetch_atp_upcoming(days_ahead: int = 2, odds_api_key: str | None = None) -> pd.DataFrame:
    """
    Fetch upcoming ATP singles matches.

    Primary source: The Odds API (works from AWS Lambda / any server IP).
    Fallback:       Sofascore (works locally, blocked by AWS IPs).

    Returns
    -------
    DataFrame with columns:
        date (date) | tournament (str) | surface (str) | round (str)
        | player1 (str) | player2 (str) | status (str)
    """
    import os
    key = odds_api_key or os.environ.get("ODDS_API_KEY", "")
    if key:
        try:
            return _fetch_upcoming_odds_api(key, days_ahead)
        except Exception as exc:
            print(f"  [Odds API fixture fetch failed: {exc}] — falling back to Sofascore")

    return _fetch_upcoming_sofascore(days_ahead)


def _fetch_upcoming_odds_api(api_key: str, days_ahead: int) -> pd.DataFrame:
    """Fetch upcoming ATP fixtures from The Odds API.

    First discovers all active ATP tournament sport-keys, then queries each
    for h2h odds. The free tier lists individual tournaments rather than a
    single generic ATP feed.
    """
    base = "https://api.the-odds-api.com/v4"

    # Step 1 — discover active ATP sport keys
    sports_resp = requests.get(f"{base}/sports/", params={"apiKey": api_key}, timeout=15)
    sports_resp.raise_for_status()
    atp_keys = [
        s["key"] for s in sports_resp.json()
        if s.get("key", "").startswith("tennis_atp") and s.get("active", False)
    ]
    if not atp_keys:
        raise RuntimeError("No active ATP sport keys found on The Odds API.")

    cutoff = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=days_ahead + 1)
    rows: list[dict] = []

    # Step 2 — query each tournament
    for sport_key in atp_keys:
        try:
            resp = requests.get(
                f"{base}/sports/{sport_key}/odds/",
                params={
                    "apiKey":     api_key,
                    "regions":    "eu",
                    "markets":    "h2h",
                    "oddsFormat": "decimal",
                    "dateFormat": "iso",
                },
                timeout=15,
            )
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
        except requests.HTTPError:
            continue

        for ev in resp.json():
            commence = ev.get("commence_time", "")
            try:
                dt = datetime.datetime.fromisoformat(commence.replace("Z", "+00:00"))
            except ValueError:
                continue
            if dt > cutoff:
                continue

            player1 = ev.get("home_team", "")
            player2 = ev.get("away_team", "")
            if not player1 or not player2:
                continue

            # Derive a clean tournament name from the sport key
            # e.g. "tennis_atp_italian_open" → "Italian Open"
            tournament = " ".join(
                w.title() for w in sport_key.replace("tennis_atp_", "").split("_")
            ) or "ATP"

            rows.append({
                "date":       dt.date(),
                "tournament": tournament,
                "surface":    "",   # Odds API doesn't provide surface
                "round":      "",
                "player1":    player1,
                "player2":    player2,
                "status":     "scheduled",
                "event_id":   ev.get("id", ""),
            })

    if not rows:
        raise RuntimeError("No upcoming ATP matches found via Odds API.")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.drop_duplicates(subset=["player1", "player2", "date"]) \
             .sort_values(["date", "tournament"]).reset_index(drop=True)


def _fetch_upcoming_sofascore(days_ahead: int) -> pd.DataFrame:
    """Fetch upcoming ATP fixtures from Sofascore (local use; blocked on Lambda)."""
    today = datetime.date.today()
    dates = [today + datetime.timedelta(d) for d in range(days_ahead + 1)]

    rows: list[dict] = []
    for day in dates:
        url = f"{_SOFASCORE_BASE}/sport/tennis/scheduled-events/{day.isoformat()}"
        resp = requests.get(url, headers=_SOFASCORE_HEADERS, timeout=20)
        resp.raise_for_status()
        try:
            events = resp.json().get("events", [])
        except ValueError:
            events = []

        for e in events:
            category = e.get("tournament", {}).get("category", {}).get("name", "")
            if category != "ATP":
                continue
            status = e.get("status", {}).get("description", "")
            if status == "Ended":
                continue
            rows.append({
                "date":       day,
                "tournament": e.get("tournament", {}).get("name", ""),
                "surface":    _sofascore_surface(e),
                "round":      e.get("roundInfo", {}).get("name", ""),
                "player1":    e.get("homeTeam", {}).get("name", ""),
                "player2":    e.get("awayTeam", {}).get("name", ""),
                "status":     status,
            })

    if not rows:
        raise RuntimeError(
            "No upcoming ATP matches found in Sofascore data. "
            "Try increasing days_ahead or check back later."
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values(["date", "tournament"]).reset_index(drop=True)


def _sofascore_surface(event: dict) -> str:
    """Extract surface string from a Sofascore event dict."""
    # groundType is available in some event dicts
    ground = event.get("groundType", "")
    if ground:
        return ground.capitalize()
    # Fall back to tournament-level info if present
    return event.get("tournament", {}).get("groundType", "")


# ---------------------------------------------------------------------------
# Name conversion utilities
# ---------------------------------------------------------------------------

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def to_internal_name(atp_full_name: str) -> str:
    """Convert 'Jannik Sinner' → 'Sinner J.' (internal dataset convention)."""
    parts = atp_full_name.strip().split()
    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]
        return f"{last} {first[0].upper()}."
    return atp_full_name


def get_live_rank_map(known_players: list[str], top_n: int = 200,
                      cache_path: "str | None" = None) -> dict[str, int]:
    """Fetch live ATP rankings. Falls back to cache_path JSON if Sofascore fails."""
    import json as _json
    from pathlib import Path as _Path

    try:
        df = fetch_atp_rankings(top_n=top_n)
        rank_map: dict[str, int] = {}
        for _, row in df.iterrows():
            internal = match_to_internal(str(row["player"]), known_players)
            if internal:
                rank_map[internal] = int(row["rank"])
        return rank_map
    except Exception:
        pass

    # Sofascore failed — try local/S3-synced cache
    cache = _Path(cache_path) if cache_path else None
    if cache and cache.exists():
        try:
            cached = _json.loads(cache.read_text())
            print(f"  [rankings] Sofascore blocked — using cached rankings ({len(cached)} players)")
            return {k: v for k, v in cached.items() if k in set(known_players)}
        except Exception:
            pass
    return {}


def _strip(t: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def match_to_internal(atp_full_name: str, known_players: list[str]) -> str | None:
    """
    Match a Sofascore/ATP full name to the internal dataset name.

    Handles compound surnames (Van Assche, Budkov Kjaer, Carreno Busta, etc.)
    and suffixes (Jr, III) by trying multiple last-name combinations.

    Strategy (in order):
    1. Single last-word conversion  e.g.  'Sinner J.'
    2. Two-word compound surname    e.g.  'Van Assche L.', 'Budkov Kjaer N.'
    3. Three-word compound surname  e.g.  'Carreno Busta P.', 'Pinnington Jones J.'
    4. Strip suffix (Jr/Sr/II/III) then retry 1-3
    5. Last-name substring search across known players (unique match only)
    """
    if not atp_full_name:
        return None

    known_lower = {p.lower(): p for p in known_players}
    parts = _strip(atp_full_name).strip().split()
    if not parts:
        return None

    first_init = parts[0][0].upper() + "."

    def _check(surname: str) -> str | None:
        candidate = f"{surname} {first_init}"
        return known_lower.get(candidate.lower())

    # Strip trailing suffix token if present
    clean_parts = parts
    if len(parts) > 1 and parts[-1].lower() in _SUFFIXES:
        clean_parts = parts[:-1]

    first_init = clean_parts[0][0].upper() + "."

    # 1. Single last word
    hit = _check(clean_parts[-1])
    if hit:
        return hit

    # 2. Two-word compound surname (e.g. "Van Assche", "Budkov Kjaer")
    if len(clean_parts) >= 3:
        compound2 = " ".join(clean_parts[-2:])
        hit = _check(compound2)
        if hit:
            return hit

    # 3. Three-word compound surname (e.g. "Carreno Busta")
    if len(clean_parts) >= 4:
        compound3 = " ".join(clean_parts[-3:])
        hit = _check(compound3)
        if hit:
            return hit

    # 4. Substring search — try last 1, 2, 3 tokens against start of known names
    for n in range(1, min(4, len(clean_parts))):
        suffix = " ".join(clean_parts[-n:]).lower()
        hits = [p for p in known_players if _strip(p).lower().startswith(suffix)]
        if len(hits) == 1:
            return hits[0]

    return None


# ---------------------------------------------------------------------------
# Sofascore player recent-match fetcher (cold-start support)
# ---------------------------------------------------------------------------

def fetch_sofascore_player_recent(player_name: str, n: int = 15) -> list[dict]:
    """
    Fetch the last *n* ATP/Challenger match results for *player_name* from
    Sofascore, for use in cold-start feature initialisation.

    player_name can be in any format (full name, "Surname F.", etc.).

    Returns a list of dicts:
        date       pd.Timestamp
        surface    str  ("Hard" | "Clay" | "Grass" | "Carpet")
        won        bool
        sets_won   int | None
        sets_lost  int | None

    Returns [] on any network or parse failure (safe to call silently).
    """
    # ── Step 1: Search Sofascore for player ID ─────────────────────────────────
    try:
        search_resp = requests.get(
            f"{_SOFASCORE_BASE}/search/all",
            headers=_SOFASCORE_HEADERS,
            params={"q": player_name},
            timeout=15,
        )
        if search_resp.status_code != 200:
            return []
        hits = search_resp.json().get("players", [])
    except Exception:
        return []

    # Pick the first tennis player found
    player_id = None
    for hit in hits:
        sport = hit.get("sport", {}).get("name", "").lower()
        if "tennis" in sport:
            player_id = hit.get("id")
            break
    if player_id is None and hits:
        player_id = hits[0].get("id")
    if not player_id:
        return []

    # ── Step 2: Fetch recent events (page 0 = most recent) ────────────────────
    try:
        events_resp = requests.get(
            f"{_SOFASCORE_BASE}/player/{player_id}/events/last/0",
            headers=_SOFASCORE_HEADERS,
            timeout=15,
        )
        if events_resp.status_code != 200:
            return []
        events = events_resp.json().get("events", [])
    except Exception:
        return []

    # ── Step 3: Parse events ───────────────────────────────────────────────────
    _SURFACE_MAP = {
        "hard": "Hard", "clay": "Clay", "grass": "Grass",
        "carpet": "Carpet", "indoor hard": "Hard",
    }
    results: list[dict] = []
    name_norm = _strip(player_name).lower()

    for ev in reversed(events):          # reversed → chronological order
        if len(results) >= n:
            break

        status = ev.get("status", {}).get("description", "")
        if status != "Ended":
            continue

        cat = ev.get("tournament", {}).get("category", {}).get("name", "")
        if cat not in ("ATP", "ATP Challenger", "ITF Men"):
            continue

        home_name = ev.get("homeTeam", {}).get("name", "")
        away_name = ev.get("awayTeam", {}).get("name", "")
        winner_code = ev.get("winnerCode")          # 1=home, 2=away

        home_norm = _strip(home_name).lower()
        away_norm = _strip(away_name).lower()

        # Fuzzy match: check if any part of the search name is in home/away
        search_parts = [p for p in name_norm.split() if len(p) > 2]
        is_home = any(p in home_norm for p in search_parts)
        is_away = any(p in away_norm for p in search_parts)

        if not is_home and not is_away:
            continue
        if is_home and is_away:
            # ambiguous — skip
            continue

        won = (winner_code == 1) if is_home else (winner_code == 2)

        # Date
        ts = ev.get("startTimestamp")
        if not ts:
            continue
        import datetime as _dt
        match_date = pd.Timestamp(_dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).date())

        # Surface
        ground = (ev.get("groundType") or
                  ev.get("tournament", {}).get("groundType") or "")
        surface = _SURFACE_MAP.get(str(ground).strip().lower(), "Hard")

        # Sets
        hs = ev.get("homeScore", {})
        as_ = ev.get("awayScore", {})
        periods = [
            (hs.get(f"period{i}"), as_.get(f"period{i}"))
            for i in range(1, 6)
            if hs.get(f"period{i}") is not None
        ]
        if periods:
            p_sets_won  = sum(1 for h, a in periods if (h > a if is_home else a > h))
            p_sets_lost = sum(1 for h, a in periods if (h < a if is_home else a < h))
        else:
            p_sets_won = p_sets_lost = None

        results.append({
            "date":      match_date,
            "surface":   surface,
            "won":       won,
            "sets_won":  p_sets_won,
            "sets_lost": p_sets_lost,
        })

    return results
