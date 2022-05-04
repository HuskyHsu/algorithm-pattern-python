# 最小生成樹

### [minimum-risk-path](https://www.lintcode.com/problem/minimum-risk-path/description)

> 地圖上有 m 條無嚮邊，每條邊 (x, y, w) 錶示位置 m 到位置 y 的權值為 w。從位置 0 到 位置 n 可能有多條路徑。我們定義一條路徑的危險值為這條路徑中所有的邊的最大權值。請問從位置 0 到 位置 n 所有路徑中最小的危險值為多少？

最小危險值為最小生成樹中 0 到 n 路徑上的最大邊權。以此題為例給出最小生成樹的兩種經典算法。

- 算法 1: [Kruskal's algorithm](<[https://en.wikipedia.org/wiki/Kruskal%27s_algorithm](https://en.wikipedia.org/wiki/Kruskal's_algorithm)>)，使用[併查集](../../data_structure/union_find.md)實現。

```Python
# Kruskal's algorithm
class Solution:
    def getMinRiskValue(self, N, M, X, Y, W):

        # Kruskal's algorithm with union-find
        parent = list(range(N + 1))
        rank = [1] * (N + 1)

        def find(x):
            if parent[parent[x]] != parent[x]:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False

            if rank[px] > rank[py]:
                parent[py] = px
            elif rank[px] < rank[py]:
                parent[px] = py
            else:
                parent[px] = py
                rank[py] += 1

            return True

        edges = sorted(zip(W, X, Y))

        for w, x, y in edges:
            if union(x, y) and find(0) == find(N): # early return without constructing MST
                return w
```

- 算法 2: [Prim's algorithm](<[https://en.wikipedia.org/wiki/Prim%27s_algorithm](https://en.wikipedia.org/wiki/Prim's_algorithm)>)，使用[堆積](../../data_structure/heap.md)實現

```Python
# Prim's algorithm
class Solution:
    def getMinRiskValue(self, N, M, X, Y, W):

        # construct graph
        adj = collections.defaultdict(list)
        for i in range(M):
            adj[X[i]].append((Y[i], W[i]))
            adj[Y[i]].append((X[i], W[i]))

        # Prim's algorithm with min heap
        MST = collections.defaultdict(list)
        min_heap = [(w, 0, v) for v, w in adj[0]]
        heapq.heapify(min_heap)

        while N not in MST:
            w, p, v = heapq.heappop(min_heap)
            if v not in MST:
                MST[p].append((v, w))
                MST[v].append((p, w))
                for n, w in adj[v]:
                    if n not in MST:
                        heapq.heappush(min_heap, (w, v, n))

        # dfs to search route from 0 to n
        dfs = [(0, None, float('-inf'))]
        while dfs:
            v, p, max_w = dfs.pop()
            for n, w in MST[v]:
                cur_max_w = max(max_w, w)
                if n == N:
                    return cur_max_w
                if n != p:
                    dfs.append((n, v, cur_max_w))
```
