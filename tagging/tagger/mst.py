"""Chu–Liu–Edmonds, scores[dependent, head], artificial ROOT at index zero."""
import numpy as np


def _cycle(heads):
    done = {0}
    for start in range(1, len(heads)):
        path, positions = [], {}
        node = start
        while node not in done:
            if node in positions:
                return path[positions[node]:]
            positions[node] = len(path)
            path.append(node)
            node = heads[node]
        done.update(path)
    return None


def _cle(scores):
    heads = scores.argmax(axis=1)
    heads[0] = 0
    cycle = _cycle(heads)
    if cycle is None:
        return heads
    inside = set(cycle)
    outside = [i for i in range(len(heads)) if i not in inside]
    index = {node: i for i, node in enumerate(outside)}
    contracted = len(outside)
    reduced = np.full((contracted + 1, contracted + 1), -np.inf)
    reduced[:contracted, :contracted] = scores[np.ix_(outside, outside)]
    # Entering a cycle replaces one selected edge; the other cycle scores are constant.
    entry, exit = {}, {}
    for node in outside:
        chosen = max(cycle, key=lambda c: scores[c, node] - scores[c, heads[c]])
        entry[node] = chosen
        reduced[contracted, index[node]] = scores[chosen, node] - scores[chosen, heads[chosen]]
        chosen = max(cycle, key=lambda c: scores[node, c])
        exit[node] = chosen
        reduced[index[node], contracted] = scores[node, chosen]
    reduced[0, :] = -np.inf
    reduced[0, 0] = 0
    parents = _cle(reduced)
    for node in outside[1:]:
        parent = parents[index[node]]
        heads[node] = exit[node] if parent == contracted else outside[parent]
    parent = outside[parents[contracted]]
    heads[entry[parent]] = parent
    return heads


def single_root_mst(arc_scores):
    """Return W 1-indexed heads for a W x (W+1) matrix; exact single-root optimum.

    Usually unrestricted CLE already has one root. Otherwise enumerate the possible
    root child and solve CLE with that constraint. This is exact, unlike root repair.
    """
    arcs = np.asarray(arc_scores, dtype=np.float64)
    n = arcs.shape[0]
    if arcs.shape != (n, n + 1):
        raise ValueError("Expected W x (W+1) arc scores")
    if not n:
        return []
    scores = np.full((n + 1, n + 1), -np.inf)
    scores[1:] = arcs
    np.fill_diagonal(scores, -np.inf)
    scores[0, 0] = 0
    if not np.isfinite(scores[1:]).any(axis=1).all():
        raise ValueError("A word has no finite head candidate")
    heads = _cle(scores)
    if np.sum(heads[1:] == 0) == 1:
        return heads[1:].tolist()
    best, best_score = None, -np.inf
    for root in range(1, n + 1):
        if not np.isfinite(scores[root, 0]):
            continue
        constrained = scores.copy()
        constrained[1:, 0] = -np.inf
        constrained[root, :] = -np.inf
        constrained[root, 0] = scores[root, 0]
        candidate = _cle(constrained)
        value = scores[np.arange(1, n + 1), candidate[1:]].sum()
        if value > best_score:
            best, best_score = candidate, value
    if best is None:
        raise ValueError("No finite single-root tree")
    return best[1:].tolist()
