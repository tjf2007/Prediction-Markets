"""Small HTTP client: polite rate limiting, retries with backoff, optional on-disk cache."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

DEFAULT_UA = "superforecasters-research/0.1"


class HttpClient:
    def __init__(
        self,
        base_url: str,
        min_interval: float = 0.25,
        cache_dir: str | os.PathLike | None = None,
        cache_ttl: float = 3600,
        max_retries: int = 4,
        timeout: float = 30,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.min_interval = min_interval
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", DEFAULT_UA)
        self._last = 0.0

    def _cache_path(self, url: str, params: dict | None) -> Path | None:
        if not self.cache_dir:
            return None
        key = url + "?" + json.dumps(params or {}, sort_keys=True)
        return self.cache_dir / (hashlib.sha1(key.encode()).hexdigest() + ".json")

    def get(self, path: str, params: dict | None = None, use_cache: bool = True, headers: dict | None = None) -> Any:
        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"
        params = {k: v for k, v in (params or {}).items() if v is not None}
        cp = self._cache_path(url, params) if use_cache else None
        if cp and cp.exists() and time.time() - cp.stat().st_mtime < self.cache_ttl:
            return json.loads(cp.read_text())

        delay = 1.0
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.session.get(url, params=params, timeout=self.timeout, headers=headers)
            except (requests.exceptions.ProxyError, requests.exceptions.SSLError):
                raise  # blocked by network policy / TLS: retrying won't help
            except requests.RequestException:
                if attempt == self.max_retries:
                    raise
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                ra = r.headers.get("Retry-After")
                time.sleep(float(ra) if ra and ra.replace(".", "").isdigit() else delay)
                delay *= 2
                continue
            r.raise_for_status()
            data = r.json()
            if cp:
                cp.parent.mkdir(parents=True, exist_ok=True)
                cp.write_text(json.dumps(data))
            return data
        raise RuntimeError("unreachable")
