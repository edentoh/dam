import re
from typing import Optional

def extract_id(s: str) -> Optional[str]:
    """
    Extracts the first occurrence of a 3-digit number from a string.
    Example: 'drawing_042.jpg' -> '042'
    """
    m = re.search(r"(\d{3})", str(s))
    return m.group(1) if m else None