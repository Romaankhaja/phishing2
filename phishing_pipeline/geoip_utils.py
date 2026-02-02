import sys
import asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import numpy as np
import geoip2.database
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def enrich_with_geoip(df, asn_db_path: str, city_db_path: str) -> Dict:
    """
    Enrich dataframe with geolocation data using MaxMind GeoIP databases.
    
    Adds ASN, country, region, and city information for each IP address.
    Silently handles missing databases and resolution failures.
    
    Args:
        df: DataFrame with 'ip_address' column
        asn_db_path: Path to GeoLite2-ASN.mmdb
        city_db_path: Path to GeoLite2-City.mmdb
    
    Returns:
        DataFrame with added columns: asn, asn_org, country, region, city
    """
    try:
        asn_reader = geoip2.database.Reader(asn_db_path)
        city_reader = geoip2.database.Reader(city_db_path)
    except FileNotFoundError as e:
        logger.warning("GeoIP database not found: %s", e)
        df["asn"] = None
        df["asn_org"] = None
        df["country"] = None
        df["region"] = None
        df["city"] = None
        return df
    except Exception as e:
        logger.error("Failed to open GeoIP databases: %s", e)
        df["asn"] = None
        df["asn_org"] = None
        df["country"] = None
        df["region"] = None
        df["city"] = None
        return df

    asn_list = []
    asn_org_list = []
    country_list = []
    region_list = []
    city_list = []

    for ip in df.get('ip_address', []):
        if not ip or (isinstance(ip, float) and np.isnan(ip)):
            asn_list.append(None)
            asn_org_list.append(None)
            country_list.append(None)
            region_list.append(None)
            city_list.append(None)
            continue
        
        # ASN lookup
        try:
            ar = asn_reader.asn(ip)
            asn_list.append(ar.autonomous_system_number)
            asn_org_list.append(ar.autonomous_system_organization)
        except geoip2.errors.GeoIP2Error as e:
            logger.debug("ASN lookup failed for %s: %s", ip, e)
            asn_list.append(None)
            asn_org_list.append(None)
        except Exception as e:
            logger.error("Unexpected error in ASN lookup for %s: %s", ip, e)
            asn_list.append(None)
            asn_org_list.append(None)
        
        # City/Country lookup
        try:
            cr = city_reader.city(ip)
            country_list.append(cr.country.iso_code)
            region_list.append(cr.subdivisions.most_specific.name if cr.subdivisions else None)
            city_list.append(cr.city.name)
        except geoip2.errors.GeoIP2Error as e:
            logger.debug("City lookup failed for %s: %s", ip, e)
            country_list.append(None)
            region_list.append(None)
            city_list.append(None)
        except Exception as e:
            logger.error("Unexpected error in city lookup for %s: %s", ip, e)
            country_list.append(None)
            region_list.append(None)
            city_list.append(None)

    # Cleanup
    try:
        asn_reader.close()
        city_reader.close()
    except Exception as e:
        logger.warning("Error closing GeoIP readers: %s", e)

    # Add columns to dataframe
    df['asn'] = asn_list
    df['asn_org'] = asn_org_list
    df['country'] = country_list
    df['region'] = region_list
    df['city'] = city_list
    
    return df
