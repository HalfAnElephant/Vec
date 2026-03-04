from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io_utils import read_json, write_json
from .models import Edge, Node


@dataclass(slots=True)
class GraphIndex:
    nodes: dict[str, Node]
    edges: list[Edge]
    adjacency: dict[str, list[str]]

    def expand_neighbors(self, seed_node_ids: list[str], hops: int = 1) -> list[str]:
        visited = set(seed_node_ids)
        frontier = set(seed_node_ids)
        for _ in range(max(1, hops)):
            nxt: set[str] = set()
            for nid in frontier:
                for nbr in self.adjacency.get(nid, []):
                    if nbr not in visited:
                        visited.add(nbr)
                        nxt.add(nbr)
            frontier = nxt
            if not frontier:
                break
        return list(visited)

    def related_edges(self, node_ids: set[str]) -> list[Edge]:
        return [e for e in self.edges if e.src_id in node_ids or e.dst_id in node_ids]



def build_graph_index(nodes: list[Node], edges: list[Edge]) -> GraphIndex:
    adjacency: dict[str, list[str]] = {n.id: [] for n in nodes}
    for edge in edges:
        adjacency.setdefault(edge.src_id, []).append(edge.dst_id)
        adjacency.setdefault(edge.dst_id, []).append(edge.src_id)
    return GraphIndex(nodes={n.id: n for n in nodes}, edges=edges, adjacency=adjacency)


def save_graph_index(path: Path, graph: GraphIndex) -> None:
    data = {
        "nodes": {nid: node.to_dict() for nid, node in graph.nodes.items()},
        "edges": [edge.to_dict() for edge in graph.edges],
        "adjacency": graph.adjacency,
    }
    write_json(path, data)


def load_graph_index(path: Path) -> GraphIndex:
    data = read_json(path)
    nodes = {nid: Node(**payload) for nid, payload in data.get("nodes", {}).items()}
    edges = [Edge(**row) for row in data.get("edges", [])]
    adjacency = {k: list(v) for k, v in data.get("adjacency", {}).items()}
    return GraphIndex(nodes=nodes, edges=edges, adjacency=adjacency)
