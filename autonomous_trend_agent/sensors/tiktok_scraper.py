"""
TikTok Trend Scraper — Stub (Phase 1, B1.3)

PLACEHOLDER — Full implementation requires deep research on:
  - Current npm packages for X-Bogus and _signature parameter generation
  - Node.js sidecar express server design for signing TikTok API URLs
  - Which TikTok internal API endpoint returns trending hashtags/sounds
  - msToken acquisition and refresh mechanism

Once the research results are received, this stub will be replaced with
the full implementation. The interface is fixed so the TrendDiscovery
orchestrator can already integrate it.

Objectives: B1.3 (TikTok Node.js Sidecar for X-Bogus signing)
"""

import logging
import subprocess
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TikTokTrend:
    """A trending topic from TikTok Creative Center."""
    hashtag: str
    post_count: int             # Posts using this hashtag
    views: int                  # Total views
    velocity: float             # Post creation rate (posts/hr)
    acceleration: float         # Delta velocity vs last period
    region: str                 # e.g. "US", "IN", "GLOBAL"
    category: str               # e.g. "Entertainment", "Education"


class TikTokScraper:
    """
    Stub for TikTok trend scraper using Node.js X-Bogus sidecar.

    AWAITING RESEARCH: The implementation will use:
        - A Node.js Express microservice running alongside the Python pipeline
        - The sidecar handles X-Bogus and _signature parameter generation
        - Python POSTs unsigned API URLs to the sidecar → receives signed URLs
        - Uses the signed URL to fetch trending data from TikTok's internal API
        - msToken is obtained via an initial session request to tiktok.com

    This stub returns an empty list so the rest of the pipeline works
    end-to-end. Replace the sidecar HTTP calls with real implementation
    once research results arrive.

    Node.js Sidecar entry point:
        autonomous_trend_agent/sensors/tiktok_sidecar/server.js
    """

    SIDECAR_URL = "http://localhost:3721"   # Port for the Node.js sidecar

    def __init__(self, sidecar_url: Optional[str] = None):
        self.sidecar_url = sidecar_url or self.SIDECAR_URL
        self._ms_token: Optional[str] = None
        logger.warning(
            "[TikTokScraper] Running in STUB mode — "
            "awaiting X-Bogus sidecar research results. Returning empty trends."
        )

    def fetch_trending(self, region: str = "US", max_results: int = 20) -> List[TikTokTrend]:
        """
        Fetch trending hashtags/sounds from TikTok Creative Center.

        Args:
            region:      ISO region code (US, IN, GB, GLOBAL)
            max_results: Max trending topics to return

        Returns:
            List of TikTokTrend (empty until stub is replaced)

        TODO (after research):
            1. Check if Node.js sidecar is running (start if not)
            2. Obtain msToken via _get_ms_token()
            3. Build TikTok Creative Center API URL
            4. POST URL to sidecar → receive X-Bogus signed URL
            5. Use signed URL to fetch trending data
            6. Parse response into TikTokTrend objects
        """
        logger.info("[TikTokScraper] STUB: returning empty TikTok trends.")
        return []

    def _check_sidecar(self) -> bool:
        """
        TODO: Verify the Node.js sidecar is running.

        Implementation:
            try:
                resp = requests.get(f"{self.sidecar_url}/health", timeout=2)
                return resp.status_code == 200
            except Exception:
                return False
        """
        return False

    def _start_sidecar(self):
        """
        TODO: Launch the Node.js sidecar as a subprocess.

        Implementation:
            sidecar_path = Path(__file__).parent / "tiktok_sidecar" / "server.js"
            subprocess.Popen(
                ["node", str(sidecar_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        """
        pass

    def _get_ms_token(self) -> Optional[str]:
        """
        TODO: Obtain msToken from TikTok.

        Implementation pattern (post-research):
            # msToken is set as a cookie on first visit to tiktok.com
            # The sidecar can extract it via a headless browser request
            resp = requests.post(f"{self.sidecar_url}/get-ms-token")
            return resp.json().get("msToken")
        """
        return None

    def _sign_url(self, url: str) -> Optional[str]:
        """
        TODO: Send URL to Node.js sidecar for X-Bogus signing.

        Implementation pattern (post-research):
            resp = requests.post(
                f"{self.sidecar_url}/sign",
                json={"url": url, "msToken": self._ms_token},
                timeout=5,
            )
            return resp.json().get("signed_url")
        """
        return None
