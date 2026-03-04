import re
from typing import Optional

def extract_id(s: str) -> Optional[str]:
    """
    Extracts the first numeric id from a string and normalizes it.
    Supports both padded and non-padded forms:
    - 'drawing_042.jpg' -> '042'
    - '1.jpeg' -> '001'
    """
    m = re.search(r"(\d+)", str(s))
    if not m:
        return None
    n = int(m.group(1))
    return f"{n:03d}"
