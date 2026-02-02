import re
import math
import socket
import ssl
import logging
import numpy as np
from urllib.parse import urlparse, ParseResult
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
import tldextract

# =====================================================================
# Configuration Constants
# =====================================================================
SSL_TIMEOUT = 3
SSL_MISSING_VALUE = -1

# =====================================================================
# Precompiled Regex Patterns (Performance Optimization)
# =====================================================================
_PROTOCOL_REGEX = re.compile(r"^https?://")
_REPEATED_DIGITS_REGEX = re.compile(r"(\d)\1")
_SPECIAL_CHARS_REGEX = re.compile(r"[^a-zA-Z0-9]")
_SPECIAL_CHARS_DOMAIN_REGEX = re.compile(r"[^a-zA-Z0-9.-]")

logger = logging.getLogger(__name__)


# =====================================================================
# URL Parsing Helper (DRY Principle)
# =====================================================================
def _normalize_and_parse_url(url: str) -> Tuple[str, ParseResult]:
    """
    Normalize URL with protocol and parse it once.
    
    Ensures all functions work with consistent URL format.
    Eliminates duplicate parsing across feature extraction functions.
    
    Args:
        url: URL string (with or without protocol)
    
    Returns:
        Tuple of (normalized_url, parsed_result)
    
    Raises:
        ValueError: If URL is invalid or empty
    """
    if not isinstance(url, str):
        raise TypeError(f"URL must be string, got {type(url).__name__}")
    
    if not url or len(url.strip()) == 0:
        raise ValueError("URL cannot be empty")
    
    url = url.strip()
    
    if len(url) > 2048:
        raise ValueError(f"URL too long: {len(url)} > 2048 characters")
    
    # Add protocol if missing
    if not _PROTOCOL_REGEX.match(url):
        url = "https://" + url
    
    try:
        parsed = urlparse(url)
        if not parsed.hostname:
            raise ValueError(f"Invalid URL - no hostname: {url}")
        return url, parsed
    except Exception as e:
        logger.error("Failed to parse URL %s: %s", url, e)
        raise ValueError(f"Invalid URL format: {url}") from e


# =====================================================================
# URL Structure Features
# =====================================================================
def extract_url_features(url: str) -> Dict[str, int | bool]:
    """
    Extract URL structural features for phishing detection.
    
    Analyzes URL length, special characters, dots, hyphens, etc.
    
    Args:
        url: URL string to analyze
    
    Returns:
        Dictionary with URL structure metrics
    """
    try:
        normalized_url, parsed = _normalize_and_parse_url(url)
    except ValueError as e:
        logger.warning("URL parsing failed: %s", e)
        return {
            "url_length": 0,
            "num_dots": 0,
            "has_repeated_digits": False,
            "num_special_chars": 0,
            "num_hyphens": 0,
            "num_slashes": 0,
            "num_underscores": 0,
            "num_question_marks": 0,
            "num_equal_signs": 0,
            "num_dollar_signs": 0,
            "num_exclamations": 0,
            "num_hashtags": 0,
            "num_percent": 0,
            "domain_length": 0,
            "num_hyphens_domain": 0,
            "has_special_chars_domain": False,
            "num_special_chars_domain": 0,
        }
    
    domain = parsed.netloc
    return {
        "url_length": len(normalized_url),
        "num_dots": normalized_url.count("."),
        "has_repeated_digits": bool(_REPEATED_DIGITS_REGEX.search(normalized_url)),
        "num_special_chars": len(_SPECIAL_CHARS_REGEX.findall(normalized_url)),
        "num_hyphens": normalized_url.count("-"),
        "num_slashes": normalized_url.count("/"),
        "num_underscores": normalized_url.count("_"),
        "num_question_marks": normalized_url.count("?"),
        "num_equal_signs": normalized_url.count("="),
        "num_dollar_signs": normalized_url.count("$"),
        "num_exclamations": normalized_url.count("!"),
        "num_hashtags": normalized_url.count("#"),
        "num_percent": normalized_url.count("%"),
        "domain_length": len(domain),
        "num_hyphens_domain": domain.count("-"),
        "has_special_chars_domain": bool(_SPECIAL_CHARS_DOMAIN_REGEX.search(domain)),
        "num_special_chars_domain": len(_SPECIAL_CHARS_DOMAIN_REGEX.findall(domain)),
    }


# =====================================================================
# Subdomain Features
# =====================================================================
def extract_subdomain_features(url: str) -> Dict[str, int | float | bool]:
    """
    Extract subdomain-specific features.
    
    Args:
        url: URL string to analyze
    
    Returns:
        Dictionary with subdomain metrics
    """
    try:
        parsed = tldextract.extract(url)
        subdomain = parsed.subdomain
        subdomains = subdomain.split(".") if subdomain else []
        
        return {
            "num_subdomains": subdomain.count(".") + 1 if subdomain else 0,
            "avg_subdomain_length": float(np.mean([len(s) for s in subdomains])) if subdomains else 0.0,
            "subdomain_length": len(subdomain),
            "subdomain_has_hyphen": "-" in subdomain if subdomain else False,
            "subdomain_has_repeated_digits": bool(_REPEATED_DIGITS_REGEX.search(subdomain)) if subdomain else False,
        }
    except Exception as e:
        logger.warning("Subdomain extraction failed for %s: %s", url, e)
        return {
            "num_subdomains": 0,
            "avg_subdomain_length": 0.0,
            "subdomain_length": 0,
            "subdomain_has_hyphen": False,
            "subdomain_has_repeated_digits": False,
        }


# =====================================================================
# Path Features
# =====================================================================
def extract_path_features(url: str) -> Dict[str, int | bool]:
    """
    Extract URL path and query features.
    
    Args:
        url: URL string to analyze
    
    Returns:
        Dictionary with path metrics
    """
    try:
        _, parsed = _normalize_and_parse_url(url)
        return {
            "path_length": len(parsed.path),
            "has_query": bool(parsed.query),
            "has_fragment": bool(parsed.fragment),
            "has_anchor": "#" in url
        }
    except ValueError as e:
        logger.warning("Path extraction failed: %s", e)
        return {
            "path_length": 0,
            "has_query": False,
            "has_fragment": False,
            "has_anchor": False
        }


# =====================================================================
# Entropy Features
# =====================================================================
def entropy_of_string(s: str) -> float:
    """
    Calculate Shannon entropy of a string.
    
    Measures randomness/disorder in character distribution.
    Higher entropy indicates more random/suspicious content.
    
    Args:
        s: Input string
    
    Returns:
        Entropy value (0.0 for empty string, 0+ for others)
    """
    if not s:
        return 0.0
    prob = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum([p * math.log2(p) for p in prob if p > 0])


def entropy_features(url: str) -> Dict[str, float]:
    """
    Extract entropy-based features from URL.
    
    Args:
        url: URL string to analyze
    
    Returns:
        Dictionary with entropy metrics
    """
    try:
        _, parsed = _normalize_and_parse_url(url)
        return {
            "entropy_url": entropy_of_string(url),
            "entropy_domain": entropy_of_string(parsed.netloc),
        }
    except ValueError as e:
        logger.warning("Entropy extraction failed: %s", e)
        return {
            "entropy_url": 0.0,
            "entropy_domain": 0.0,
        }


# =====================================================================
# IP Address Extraction
# =====================================================================
def get_ip_address(url: str) -> Optional[str]:
    """
    Resolve hostname to IP address.
    
    Args:
        url: URL string to resolve
    
    Returns:
        IP address string or None if resolution fails
    """
    try:
        _, parsed = _normalize_and_parse_url(url)
        hostname = parsed.hostname
        if not hostname:
            return None
        return socket.gethostbyname(hostname)
    except socket.gaierror as e:
        logger.debug("DNS resolution failed for %s: %s", url, e)
        return None
    except socket.timeout:
        logger.debug("DNS lookup timeout for %s", url)
        return None
    except ValueError:
        logger.debug("Invalid URL format: %s", url)
        return None
    except Exception as e:
        logger.error("Unexpected error resolving %s: %s", url, e)
        return None


# =====================================================================
# SSL/TLS Certificate Features
# =====================================================================
def _parse_ssl_date(date_str: str) -> datetime:
    """
    Parse SSL certificate date string (GMT/UTC).
    
    Handles various formats and adds UTC timezone info.
    
    Args:
        date_str: SSL certificate date string (e.g., 'Jan 15 00:00:00 2024 GMT')
    
    Returns:
        Timezone-aware datetime object (UTC)
    
    Raises:
        ValueError: If date format cannot be parsed
    """
    try:
        # Remove timezone suffix (always UTC in SSL certs)
        date_parts = date_str.split()
        date_clean = ' '.join(date_parts[:-1])  # Remove 'GMT' or 'UTC'
        
        dt = datetime.strptime(date_clean, '%b %d %H:%M:%S %Y')
        # Add UTC timezone awareness
        return dt.replace(tzinfo=timezone.utc)
    except ValueError as e:
        logger.error("Failed to parse SSL date '%s': %s", date_str, e)
        raise


def ssl_features(url: str) -> Dict[str, int | str | None]:
    """
    Extract SSL/TLS certificate features.
    
    Checks certificate presence, validity, expiry, and issuer information.
    
    Args:
        url: URL string to check
    
    Returns:
        Dictionary with SSL certificate metrics
    """
    features = {
        "ssl_present": 0,
        "ssl_valid": 0,
        "ssl_days_to_expiry": SSL_MISSING_VALUE,
        "ssl_issuer": None
    }
    
    try:
        _, parsed = _normalize_and_parse_url(url)
        hostname = parsed.hostname
        if not hostname:
            return features

        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=SSL_TIMEOUT) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                if not cert:
                    return features
                
                features["ssl_present"] = 1
                
                try:
                    not_before = _parse_ssl_date(cert['notBefore'])
                    not_after = _parse_ssl_date(cert['notAfter'])
                    now = datetime.now(timezone.utc)
                    
                    features["ssl_valid"] = int(not_before <= now <= not_after)
                    features["ssl_days_to_expiry"] = max(0, (not_after - now).days)
                except (ValueError, KeyError) as e:
                    logger.error("SSL date parsing failed: %s", e)
                    features["ssl_days_to_expiry"] = SSL_MISSING_VALUE
                
                try:
                    issuer = dict(x[0] for x in cert.get('issuer', []))
                    features["ssl_issuer"] = issuer.get("O", issuer.get("organizationName", None))
                except Exception as e:
                    logger.warning("Failed to extract issuer from certificate: %s", e)
    
    except socket.timeout:
        logger.debug("SSL connection timeout for %s", url)
    except socket.gaierror as e:
        logger.debug("DNS resolution failed for %s: %s", url, e)
    except OSError as e:
        logger.debug("Network error for %s: %s", url, e)
    except ssl.SSLError as e:
        logger.debug("SSL error for %s: %s", url, e)
    except ValueError:
        logger.debug("Invalid URL format: %s", url)
    except Exception as e:
        logger.error("Unexpected error in ssl_features for %s: %s", url, e)
    
    return features
