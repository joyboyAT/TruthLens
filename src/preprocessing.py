from typing import Any, Dict, List
import pandas as pd
from bs4 import BeautifulSoup

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or '', 'html.parser')
    return soup.get_text(' ', strip=True)

def build_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=['url','status','html','error'])
    return pd.DataFrame.from_records(records)
