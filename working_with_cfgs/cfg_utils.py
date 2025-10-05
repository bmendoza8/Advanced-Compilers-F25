def get_path_lengths(cfg, entry):
    from collections import deque
    dist = {entry: 0}
    queue = deque([entry])
    while queue:
        node = queue.popleft()
        for succ in cfg.get(node, []):
            if succ not in dist:
                dist[succ] = dist[node] + 1
                queue.append(succ)
    return dist


def reverse_postorder(cfg, entry):
    visited = set()
    order = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for succ in cfg.get(node, []):
            dfs(succ)
        order.append(node)

    dfs(entry)
    order.reverse()
    return order


def find_back_edges(cfg, entry):
    visited, stack, back_edges = set(), set(), []

    def dfs(u):
        visited.add(u)
        stack.add(u)
        for v in cfg.get(u, []):
            if v in stack:
                back_edges.append((u, v))
            elif v not in visited:
                dfs(v)
        stack.remove(u)

    dfs(entry)
    return back_edges


def is_reducible(cfg, entry):
    back_edges = find_back_edges(cfg, entry)
    nodes_in_cycles = {v for (_, v) in back_edges}
    return len(nodes_in_cycles) == len(back_edges)
