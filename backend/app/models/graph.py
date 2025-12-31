from pydantic import BaseModel
from typing import List, Optional

class Node(BaseModel):
    id: str
    label: str  # e.g., "Project", "Library"
    name: str
    risk_score: float = 0.0
    description: Optional[str] = None
    history: Optional[List[float]] = None # Historical risk scores (last 6 months)

class Edge(BaseModel):
    source: str
    target: str
    relation: str  # e.g., "DEPENDS_ON"

class GraphData(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
