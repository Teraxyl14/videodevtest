"""
Twitter/X Trend Scraper (Phase 1, B1.2)

Full implementation using curl_cffi TLS fingerprint spoofing,
guest token provisioning, and GraphQL SearchTimeline endpoint.

Architecture based on research (Social Media Scraping_ 2026.txt):
  - curl_cffi with impersonate="chrome124" for JA3/TLS bypass
  - Static Bearer token + dynamic guest token from activate.json
  - X-Client-Transaction-Id reconstructed in pure Python (SHA-256 + XOR)
  - SearchTimeline GraphQL with full 2026 feature flags
  - Velocity signals from result.views.count + result.legacy.*_count

Rate-limiting rules (critical):
  - Guest tokens are IP-BOUND — never rotate proxy mid-session
  - Session boundary = every 2-4 hours, then get fresh token + IP
  - queryId rotates every 2-4 weeks — dynamically fetched from JS bundles

Objectives: B1.2 (Twitter/X Guest Token Scraping, curl_cffi + TLS spoofing)
"""

import os
import re
import json
import time
import math
import hashlib
import base64
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict

from curl_cffi import requests as cffi_requests

logger = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────────────────────

# Static Bearer token embedded in X's frontend JavaScript (as of early 2026).
# Rotates very rarely — update here if requests start returning 401.
STATIC_BEARER = (
    "AAAAAAAAAAAAAAAAAAAAAFXzAwAAAAAAMHCxpeSDG1gLNLghVe8d74hl6k4%3D"
    "RUMF4xAQLsbeBhTSRrCiQpJtxoGWeyHrDb5te2jpGskWDFW82F"
)

# Default User-Agent — must match across all requests in the same session
DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Seconds since the April 2023 epoch used by X's Transaction ID algorithm
X_EPOCH = 1682924400

# GraphQL feature flags required by the 2026 SearchTimeline endpoint
SEARCH_FEATURES = {
    "rweb_lists_timeline_redesign_enabled": True,
    "responsive_web_graphql_exclude_directive_enabled": True,
    "verified_phone_label_enabled": False,
    "creator_subscriptions_tweet_preview_api_enabled": True,
    "responsive_web_graphql_timeline_navigation_enabled": True,
    "tweetypie_unmention_optimization_enabled": True,
    "vibe_api_enabled": True,
    "responsive_web_edit_tweet_api_enabled": True,
    "view_counts_everywhere_api_enabled": True,
    "longform_notetweets_consumption_enabled": True,
}

# Known good queryId for SearchTimeline — dynamically refreshed at boot
# Rotates every 2-4 weeks; if requests fail with 400, run _fetch_query_id()
SEARCH_TIMELINE_QUERY_ID = "ukD99BOU37OlcBLMSDFRvQ"

# Trending topics to query when no explicit keywords are provided
DEFAULT_TREND_QUERIES = [
    "trending today",
    "viral video",
    "breaking news",
    "what is happening",
]


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class TwitterTrend:
    """A trending topic with velocity signals from Twitter/X."""
    topic: str
    tweet_volume: int           # Estimated tweets containing this topic
    impressions: int            # Total view impressions collected
    sentiment_score: float      # -1.0 to +1.0 (positive ratios)
    velocity: float             # Engagements per hour
    acceleration: float         # Delta velocity vs last poll
    sample_tweets: List[str] = field(default_factory=list)


# ─── X-Client-Transaction-Id generation ──────────────────────────────────────

def _generate_transaction_id(method: str, path: str) -> str:
    """
    Reconstructs the X-Client-Transaction-Id payload in pure Python.

    Algorithm (reverse-engineered from X frontend JS):
      1. Compute seconds since X epoch (April 2023)
      2. SHA-256 of "{method}!{path}!{timestamp}obfiowerehiring{key}"
      3. XOR-obfuscate with random salt byte
      4. Base64-encode (no padding)

    Source: Medium — 'X Login Flow Reconstruction Episode 1'
    """
    time_now = math.floor((time.time() * 1000 - X_EPOCH * 1000) / 1000)
    time_bytes = [(time_now >> (i * 8)) & 0xFF for i in range(4)]

    animation_key = "default_key"
    hash_input = f"{method}!{path}!{time_now}obfiowerehiring{animation_key}"
    hash_val = hashlib.sha256(hash_input.encode("utf-8")).digest()

    key_bytes = [0x00] * 16
    random_num = random.randint(0, 255)
    bytes_arr = key_bytes + time_bytes + list(hash_val)[:16]
    out = bytearray([random_num] + [b ^ random_num for b in bytes_arr])

    return base64.b64encode(out).decode("utf-8").rstrip("=")


# ─── Token Manager ────────────────────────────────────────────────────────────

class XTokenManager:
    """
    Manages the acquisition and rotation of X.com guest tokens.

    IMPORTANT: Guest tokens are IP-bound. Never rotate proxy IP mid-session.
    Rotate the proxy AND get a fresh token together at session boundary.
    """

    def __init__(self, proxy_url: Optional[str] = None):
        self.proxy_url = proxy_url
        proxies = {}
        if proxy_url:
            proxies = {"http": proxy_url, "https": proxy_url}

        self.session = cffi_requests.Session(
            impersonate="chrome124",
            proxies=proxies,
            timeout=20,
        )
        self.guest_token: Optional[str] = None
        self._token_born_at: float = 0.0

    def get_token(self, force_refresh: bool = False) -> str:
        """Return a valid guest token, refreshing if expired (>2 hrs)."""
        age_hrs = (time.time() - self._token_born_at) / 3600
        if force_refresh or not self.guest_token or age_hrs > 2.0:
            self.guest_token = self._activate()
            self._token_born_at = time.time()
        return self.guest_token

    def _activate(self) -> str:
        """POST to activate.json to get a fresh guest token."""
        headers = {
            "User-Agent": DEFAULT_UA,
            "Authorization": f"Bearer {STATIC_BEARER}",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "application/json",
        }
        resp = self.session.post(
            "https://api.x.com/1.1/guest/activate.json",
            headers=headers,
        )
        resp.raise_for_status()
        token = resp.json().get("guest_token")
        if not token:
            raise RuntimeError("activate.json did not return guest_token")
        logger.info(f"[XTokenManager] New guest token provisioned: {token[:8]}…")
        return token


# ─── QueryId fetcher ──────────────────────────────────────────────────────────

def _fetch_query_id(session: cffi_requests.Session) -> str:
    """
    Dynamically scrape the current SearchTimeline queryId from X's JS bundles.
    Falls back to the hardcoded constant if scraping fails.

    X rotates queryIds every 2-4 weeks to break scrapers.
    """
    try:
        resp = session.get(
            "https://x.com/",
            headers={"User-Agent": DEFAULT_UA},
            timeout=15,
        )
        # Find JS bundle URLs in the page
        bundle_urls = re.findall(r'"(https://abs\.twimg\.com/responsive-web/[^"]+\.js)"', resp.text)

        for url in bundle_urls[:8]:  # Scan first 8 bundles only
            try:
                js = session.get(url, headers={"User-Agent": DEFAULT_UA}, timeout=10).text
                # Pattern: queryId:"<id>",operationName:"SearchTimeline"
                match = re.search(r'queryId:"([^"]+)",operationName:"SearchTimeline"', js)
                if match:
                    qid = match.group(1)
                    logger.info(f"[TwitterScraper] Found live queryId: {qid}")
                    return qid
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"[TwitterScraper] queryId scrape failed ({e}), using fallback.")

    return SEARCH_TIMELINE_QUERY_ID


# ─── GraphQL helpers ──────────────────────────────────────────────────────────

def _search_timeline(
    session: cffi_requests.Session,
    query_id: str,
    guest_token: str,
    search_term: str,
    cursor: Optional[str] = None,
) -> dict:
    """Execute a SearchTimeline GraphQL query."""
    path = f"/i/api/graphql/{query_id}/SearchTimeline"
    url = f"https://x.com{path}"

    variables: Dict = {
        "rawQuery": search_term,
        "count": 20,
        "querySource": "typed_query",
        "product": "Top",
    }
    if cursor:
        variables["cursor"] = cursor

    params = {
        "variables": json.dumps(variables, separators=(",", ":")),
        "features": json.dumps(SEARCH_FEATURES, separators=(",", ":")),
    }

    headers = {
        "User-Agent": DEFAULT_UA,
        "Authorization": f"Bearer {STATIC_BEARER}",
        "x-guest-token": guest_token,
        "x-twitter-active-user": "yes",
        "x-twitter-client-language": "en",
        "x-client-transaction-id": _generate_transaction_id("GET", path),
        "Referer": "https://x.com/search",
        "Content-Type": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }

    resp = session.get(url, params=params, headers=headers, timeout=20)

    if resp.status_code == 429:
        raise RuntimeError("RateLimitExceeded: rotate token and proxy")
    if resp.status_code == 400:
        raise RuntimeError("Bad request — queryId may be stale, trigger _fetch_query_id()")

    resp.raise_for_status()
    return resp.json()


def _parse_timeline(json_response: dict) -> Tuple[List[dict], Optional[str]]:
    """
    Parse the deeply nested GraphQL TimelineAddEntries response.
    Returns list of tweet dicts + optional pagination cursor.

    Data paths (2026):
        views:    result.views.count
        retweets: result.legacy.retweet_count
        likes:    result.legacy.favorite_count
        replies:  result.legacy.reply_count
    """
    tweets: List[dict] = []
    next_cursor: Optional[str] = None

    try:
        instructions = (
            json_response
            .get("data", {})
            .get("search_by_raw_query", {})
            .get("search_timeline", {})
            .get("timeline", {})
            .get("instructions", [])
        )

        entries: List[dict] = []
        for inst in instructions:
            if inst.get("type") == "TimelineAddEntries":
                entries = inst.get("entries", [])
            elif (
                inst.get("type") == "TimelineReplaceEntry"
                and inst.get("entry", {}).get("entryId") == "sq-cursor-bottom"
            ):
                next_cursor = inst["entry"]["content"].get("value")

        for entry in entries:
            entry_id = str(entry.get("entryId", ""))

            # Grab pagination cursor
            if entry_id.startswith(("sq-cursor-bottom", "cursor-bottom")):
                next_cursor = (
                    entry.get("content", {}).get("itemContent", {}).get("value")
                    or entry.get("content", {}).get("value")
                )
                continue

            if not entry_id.startswith("tweet-"):
                continue  # Skip ads, user cards, etc.

            try:
                result = (
                    entry["content"]["itemContent"]["tweet_results"]["result"]
                )
                # Handle quoted/retweeted nesting
                if "tweet" in result:
                    result = result["tweet"]

                legacy = result.get("legacy", {})
                views_str = result.get("views", {}).get("count", "0")

                retweets = int(legacy.get("retweet_count", 0))
                likes = int(legacy.get("favorite_count", 0))
                replies = int(legacy.get("reply_count", 0))
                views = int(views_str) if views_str else 0

                tweets.append({
                    "tweet_id": legacy.get("id_str", ""),
                    "text": legacy.get("full_text", ""),
                    "created_at": legacy.get("created_at", ""),
                    "retweets": retweets,
                    "likes": likes,
                    "replies": replies,
                    "views": views,
                    "engagement": retweets + likes + replies,
                })
            except KeyError:
                continue  # Silently skip deleted/malformed tweets

    except KeyError as e:
        logger.error(f"[TwitterScraper] Timeline JSON traversal failed: {e}")

    return tweets, next_cursor


# ─── Main scraper ─────────────────────────────────────────────────────────────

class TwitterScraper:
    """
    Twitter/X trend scraper.

    Fetches trending topics using GraphQL SearchTimeline + curl_cffi TLS spoofing.

    Usage:
        scraper = TwitterScraper()
        trends = scraper.fetch_trending()
    """

    def __init__(
        self,
        bearer_token: Optional[str] = None,
        proxy_url: Optional[str] = None,
    ):
        self._proxy_url = proxy_url or os.getenv("PROXY_URL")  # Optional residential proxy
        self._token_mgr = XTokenManager(proxy_url=self._proxy_url)
        self._query_id: Optional[str] = None
        self._prev_stats: Dict[str, Dict] = {}
        logger.info("[TwitterScraper] Initialized.")

    def fetch_trending(
        self,
        woeid: int = 1,
        queries: Optional[List[str]] = None,
        max_pages: int = 2,
    ) -> List[TwitterTrend]:
        """
        Fetch trending topics and compute velocity signals.

        Args:
            woeid:      Where On Earth ID (unused directly — Twitter deprecated
                        trends/place.json; we query popular search terms instead)
            queries:    Search terms to probe. Defaults to DEFAULT_TREND_QUERIES.
            max_pages:  Pages to fetch per query (20 tweets/page, max recommended: 2)

        Returns:
            List of TwitterTrend sorted by velocity descending.
        """
        queries = queries or DEFAULT_TREND_QUERIES

        try:
            guest_token = self._token_mgr.get_token()
        except Exception as e:
            logger.error(f"[TwitterScraper] Failed to get guest token: {e}")
            return []

        # Lazy queryId init — scrape from JS bundles once per session
        if not self._query_id:
            self._query_id = _fetch_query_id(self._token_mgr.session)

        all_tweet_groups: Dict[str, List[dict]] = {}

        for query in queries:
            tweets: List[dict] = []
            cursor: Optional[str] = None

            for page in range(max_pages):
                try:
                    raw = _search_timeline(
                        session=self._token_mgr.session,
                        query_id=self._query_id,
                        guest_token=guest_token,
                        search_term=query,
                        cursor=cursor,
                    )
                    page_tweets, cursor = _parse_timeline(raw)
                    tweets.extend(page_tweets)

                    if not cursor or not page_tweets:
                        break

                    time.sleep(random.uniform(1.0, 2.5))  # Polite pacing

                except RuntimeError as e:
                    err = str(e)
                    if "RateLimitExceeded" in err:
                        logger.warning(f"[TwitterScraper] Rate limit on '{query}'. Pausing 60s.")
                        time.sleep(60)
                        guest_token = self._token_mgr.get_token(force_refresh=True)
                    elif "queryId" in err:
                        logger.warning("[TwitterScraper] queryId stale — refreshing.")
                        self._query_id = _fetch_query_id(self._token_mgr.session)
                    break
                except Exception as e:
                    logger.error(f"[TwitterScraper] Error on query '{query}': {e}")
                    break

            if tweets:
                all_tweet_groups[query] = tweets

        return self._aggregate(all_tweet_groups)

    def _aggregate(self, groups: Dict[str, List[dict]]) -> List[TwitterTrend]:
        """Convert raw tweet groups into TwitterTrend objects with velocity."""
        results: List[TwitterTrend] = []

        for query, tweets in groups.items():
            if not tweets:
                continue

            total_views = sum(t["views"] for t in tweets)
            total_engagement = sum(t["engagement"] for t in tweets)
            tweet_count = len(tweets)

            # Estimate velocity: total engagement per hour based on tweet ages
            now = time.time()
            hours_span = 1.0
            try:
                oldest = min(
                    datetime.strptime(t["created_at"], "%a %b %d %H:%M:%S +0000 %Y")
                    .replace(tzinfo=timezone.utc).timestamp()
                    for t in tweets if t.get("created_at")
                )
                hours_span = max(0.1, (now - oldest) / 3600)
            except Exception:
                pass

            velocity = total_engagement / hours_span

            # Acceleration vs previous poll
            prev = self._prev_stats.get(query, {})
            acceleration = velocity - prev.get("velocity", 0.0)

            # Sentiment: likes / (likes + replies) as a proxy
            total_likes = sum(t["likes"] for t in tweets)
            total_replies = sum(t["replies"] for t in tweets)
            denom = total_likes + total_replies
            sentiment = (total_likes / denom - 0.5) * 2 if denom > 0 else 0.0

            sample_tweets = [t["text"][:140] for t in tweets[:3] if t.get("text")]

            results.append(TwitterTrend(
                topic=query,
                tweet_volume=tweet_count,
                impressions=total_views,
                sentiment_score=round(sentiment, 3),
                velocity=round(velocity, 1),
                acceleration=round(acceleration, 1),
                sample_tweets=sample_tweets,
            ))

            # Save for next poll
            self._prev_stats[query] = {"velocity": velocity, "snapshot": now}

        results.sort(key=lambda t: t.velocity, reverse=True)
        logger.info(f"[TwitterScraper] Aggregated {len(results)} trend topics.")
        return results
