# phishing_pipeline/rate_limiter.py
"""
Rate limiter utility for throttling WHOIS and other network requests.
Ensures compliance with server rate limits to prevent IP blocking.
"""

import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    A time-based rate limiter that enforces a minimum delay between operations.
    
    Uses a simple time-tracking approach to ensure requests are evenly spaced.
    Thread-safe for async operations via asyncio.Lock.
    
    Args:
        requests_per_minute: Maximum requests allowed per minute (default: 20)
    
    Example:
        limiter = RateLimiter(requests_per_minute=20)  # 3 seconds between requests
        
        for domain in domains:
            await limiter.acquire()
            result = whois.whois(domain)
    """
    
    def __init__(self, requests_per_minute: int = 20):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        
        self.requests_per_minute = requests_per_minute
        self.delay = 60.0 / requests_per_minute  # Seconds between requests
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()
        
        logger.info(
            f"🚦 RateLimiter initialized: {requests_per_minute} req/min "
            f"(delay: {self.delay:.2f}s)"
        )
    
    async def acquire(self):
        """
        Wait until we're allowed to make the next request.
        
        This method blocks (asynchronously) until enough time has passed
        since the last request. It's safe to call from multiple coroutines.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_request_time
            wait_time = self.delay - elapsed
            
            if wait_time > 0:
                logger.debug(f"Rate limiter: waiting {wait_time:.2f}s before next request")
                await asyncio.sleep(wait_time)
            
            self.last_request_time = time.monotonic()
    
    def reset(self):
        """Reset the rate limiter (useful for testing or restarts)."""
        self.last_request_time = 0.0
        logger.debug("Rate limiter reset")
