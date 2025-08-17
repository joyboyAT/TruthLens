from typing import Dict, List, Optional, TypedDict

import requests

class CollectedRecord(TypedDict, total=False):
    url: str
    status: Optional[int]
    html: Optional[str]
    error: Optional[str]

def fetch_url(url: str, timeout: int = 15, headers: Optional[Dict[str, str]] = None) -> CollectedRecord:
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        return {'url': url, 'status': resp.status_code, 'html': resp.text}
    except requests.RequestException as exc:
        return {'url': url, 'error': str(exc)}

def collect_from_urls(urls: List[str], timeout: int = 15, headers: Optional[Dict[str, str]] = None) -> List[CollectedRecord]:
    return [fetch_url(u, timeout=timeout, headers=headers) for u in urls]
