from typing import Optional, Collection, Union, List, Any

# Mock types for fastai compatibility
Floats = Union[float, Collection[float]]

def listify(p: Any = None, q: Any = None) -> List[Any]:
    """Convert input into a list."""
    if p is None: 
        return []
    if isinstance(p, list): 
        return p
    if isinstance(p, (tuple, set)): 
        return list(p)
    return [p]
