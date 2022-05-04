# 最短路徑問題

## BFS

在處理不帶權圖的最短路徑問題時可以使用 BFS。

### [walls-and-gates](https://leetcode.com/problems/walls-and-gates/)

> 給定一個二維矩陣，矩陣中元素 -1 錶示牆或是障礙物，0 錶示一扇門，INF (2147483647) 錶示一個空的房間。你要給每個空房間位上填上該房間到最近門的距離，如果無法到達門，則填 INF 即可。

- 思路：典型的多源最短路徑問題，將所有源作為 BFS 的第一層即可

```Python
inf = 2147483647

class Solution:
    def wallsAndGates(self, rooms: List[List[int]]) -> None:
        """
        Do not return anything, modify rooms in-place instead.
        """

        if not rooms or not rooms[0]:
            return

        M, N = len(rooms), len(rooms[0])

        bfs = collections.deque([])

        for i in range(M):
            for j in range(N):
                if rooms[i][j] == 0:
                    bfs.append((i, j))

        dist = 1
        while bfs:
            num_level = len(bfs)
            for _ in range(num_level):
                r, c = bfs.popleft()

                if r - 1 >= 0 and rooms[r - 1][c] == inf:
                    rooms[r - 1][c] = dist
                    bfs.append((r - 1, c))

                if r + 1 < M and rooms[r + 1][c] == inf:
                    rooms[r + 1][c] = dist
                    bfs.append((r + 1, c))

                if c - 1 >= 0 and rooms[r][c - 1] == inf:
                    rooms[r][c - 1] = dist
                    bfs.append((r, c - 1))

                if c + 1 < N and rooms[r][c + 1] == inf:
                    rooms[r][c + 1] = dist
                    bfs.append((r, c + 1))

            dist += 1

        return
```

### [shortest-bridge](https://leetcode.com/problems/shortest-bridge/)

> 在給定的 01 矩陣 A 中，存在兩座島 (島是由四麵相連的 1 形成的一個連通分量)。現在，我們可以將 0 變為 1，以使兩座島連接起來，變成一座島。返回必須翻轉的 0 的最小數目。

- 思路：DFS 遍歷連通分量找邊界，從邊界開始 BFS 找最短路徑

```Python
class Solution:
    def shortestBridge(self, A: List[List[int]]) -> int:

        M, N = len(A), len(A[0])
        neighors = ((-1, 0), (1, 0), (0, -1), (0, 1))

        dfs = []
        bfs = collections.deque([])

        for i in range(M):
            for j in range(N):
                if A[i][j] == 1: # start from a 1
                    dfs.append((i, j))
                    break
            if dfs:
                break

        while dfs:
            r, c = dfs.pop()
            if A[r][c] == 1:
                A[r][c] = -1

                for dr, dc in neighors:
                    nr, nc = r + dr, c + dc
                    if 0<= nr < M and 0 <= nc < N:
                        if A[nr][nc] == 0: # meet and edge
                            A[nr][nc] = -2
                            bfs.append((nr, nc))
                        elif A[nr][nc] == 1:
                            dfs.append((nr, nc))

        flip = 1
        while bfs:
            num_level = len(bfs)
            for _ in range(num_level):
                r, c = bfs.popleft()

                for dr, dc in neighors:
                    nr, nc = r + dr, c + dc
                    if 0<= nr < M and 0 <= nc < N:
                        if A[nr][nc] == 0:
                            A[nr][nc] = -2
                            bfs.append((nr, nc))
                        elif A[nr][nc] == 1:
                            return flip
            flip += 1
```

## Dijkstra's Algorithm

用於求解單源最短路徑問題。思想是 greedy 構造 shortest path tree (SPT)，每次將當前距離源點最短的不在 SPT 中的結點加入 SPT，與構造最小生成樹 (MST) 的 Prim's algorithm 非常相似。可以用 priority queue (heap) 實現。

### [network-delay-time](https://leetcode.com/problems/network-delay-time/)

- 標準的單源最短路徑問題，使用樸素的 Dijikstra 算法即可。

```Python
class Solution:
    def networkDelayTime(self, times: List[List[int]], N: int, K: int) -> int:

        # construct graph
        graph_neighbor = collections.defaultdict(list)
        for s, e, t in times:
            graph_neighbor[s].append((e, t))

        # Dijkstra
        SPT = {}
        min_heap = [(0, K)]

        while min_heap:
            delay, node = heapq.heappop(min_heap)
            if node not in SPT:
                SPT[node] = delay
                for n, d in graph_neighbor[node]:
                    if n not in SPT:
                        heapq.heappush(min_heap, (d + delay, n))

        return max(SPT.values()) if len(SPT) == N else -1
```

### [cheapest-flights-within-k-stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/)

- 在標準的單源最短路徑問題上限製了路徑的邊數，因此需要同時維護當前 SPT 內每個結點最短路徑的邊數，當遇到邊數更小的路徑 (邊權和可以更大) 時結點需要重新入堆，以更新後繼在邊數上限內冇達到的結點。

```Python
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, K: int) -> int:

        # construct graph
        graph_neighbor = collections.defaultdict(list)
        for s, e, p in flights:
            graph_neighbor[s].append((e, p))

        # modified Dijkstra
        prices, steps = {}, {}
        min_heap = [(0, 0, src)]

        while len(min_heap) > 0:
            price, step, node = heapq.heappop(min_heap)

            if node == dst: # early return
                return price

            if node not in prices:
                prices[node] = price

            steps[node] = step
            if step <= K:
                step += 1
                for n, p in graph_neighbor[node]:
                    if n not in prices or step < steps[n]:
                        heapq.heappush(min_heap, (p + price, step, n))

        return -1
```

## 補充

## Bidrectional BFS

當求點對點的最短路徑時，BFS 遍歷結點數目隨路徑長度呈指數增長，為縮小遍歷結點數目可以考慮從起點 BFS 的同時從終點也做 BFS，當路徑相遇時得到最短路徑。

### [word-ladder](https://leetcode.com/problems/word-ladder/)

```Python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:

        N, K = len(wordList), len(beginWord)

        find_end = False
        for i in range(N):
            if wordList[i] == endWord:
                find_end = True
                break

        if not find_end:
            return 0

        wordList.append(beginWord)
        N += 1

        # clustering nodes for efficiency compare to adjacent list
        cluster = collections.defaultdict(list)
        for i in range(N):
            node = wordList[i]
            for j in range(K):
                cluster[node[:j] + '*' + node[j + 1:]].append(node)

        # bidirectional BFS
        visited_start, visited_end = set([beginWord]), set([endWord])
        bfs_start, bfs_end = collections.deque([beginWord]), collections.deque([endWord])
        step = 2
        while bfs_start and bfs_end:

            # start
            num_level = len(bfs_start)
            while num_level > 0:
                node = bfs_start.popleft()
                for j in range(K):
                    key = node[:j] + '*' + node[j + 1:]
                    for n in cluster[key]:
                        if n in visited_end: # if meet, route from start larger by 1 than route from end
                            return step * 2 - 2
                        if n not in visited_start:
                            visited_start.add(n)
                            bfs_start.append(n)
                num_level -= 1

            # end
            num_level = len(bfs_end)
            while num_level > 0:
                node = bfs_end.popleft()
                for j in range(K):
                    key = node[:j] + '*' + node[j + 1:]
                    for n in cluster[key]:
                        if n in visited_start: # if meet, route from start equals route from end
                            return step * 2 - 1
                        if n not in visited_end:
                            visited_end.add(n)
                            bfs_end.append(n)
                num_level -= 1
            step += 1

        return 0
```

## A\* Algorithm

當需要求解有目標的最短路徑問題時，BFS 或 Dijkstra's algorithm 可能會搜索過多冗餘的其他目標從而降低搜索效率，此時可以考慮使用 A* algorithm。原理不展開，有興趣可以自行搜索。實現上和 Dijkstra’s algorithm 非常相似，隻是優先級需要加上一個到目標點距離的估值，這個估值嚴格小於等於真正的最短距離時保證得到最優解。當 A* algorithm 中的距離估值為 0 時 退化為 BFS 或 Dijkstra’s algorithm。

### [sliding-puzzle](https://leetcode.com/problems/sliding-puzzle)

- 方法 1：BFS。為了方便對比 A\* 算法寫成了與其相似的形式。

```Python
class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:

        next_move = {
            0: [1, 3],
            1: [0, 2, 4],
            2: [1, 5],
            3: [0, 4],
            4: [1, 3, 5],
            5: [2, 4]
        }

        start = tuple(itertools.chain(*board))
        target = (1, 2, 3, 4, 5, 0)
        target_wrong = (1, 2, 3, 5, 4, 0)

        SPT = set()
        bfs = collections.deque([(0, start, start.index(0))])

        while bfs:
            step, state, idx0 = bfs.popleft()

            if state == target:
                return step

            if state == target_wrong:
                return -1

            if state not in SPT:
                SPT.add(state)

                for next_step in next_move[idx0]:
                    next_state = list(state)
                    next_state[idx0], next_state[next_step] = next_state[next_step], next_state[idx0]
                    next_state = tuple(next_state)

                    if next_state not in SPT:
                        bfs.append((step + 1, next_state, next_step))
        return -1
```

- 方法 2：A\* algorithm

```Python
class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:

        next_move = {
            0: [1, 3],
            1: [0, 2, 4],
            2: [1, 5],
            3: [0, 4],
            4: [1, 3, 5],
            5: [2, 4]
        }

        start = tuple(itertools.chain(*board))
        target, target_idx = (1, 2, 3, 4, 5, 0), (5, 0, 1, 2, 3, 4)
        target_wrong = (1, 2, 3, 5, 4, 0)

        @functools.lru_cache(maxsize=None)
        def taxicab_dist(x, y):
            return abs(x // 3 - y // 3) + abs(x % 3 - y % 3)

        def taxicab_sum(state, t_idx):
            result = 0
            for i, num in enumerate(state):
                result += taxicab_dist(i, t_idx[num])
            return result

        SPT = set()
        min_heap = [(0 + taxicab_sum(start, target_idx), 0, start, start.index(0))]

        while min_heap:
            cur_cost, step, state, idx0 = heapq.heappop(min_heap)

            if state == target:
                return step

            if state == target_wrong:
                return -1

            if state not in SPT:
                SPT.add(state)

                for next_step in next_move[idx0]:
                    next_state = list(state)
                    next_state[idx0], next_state[next_step] = next_state[next_step], next_state[idx0]
                    next_state = tuple(next_state)
                    next_cost = step + 1 + taxicab_sum(next_state, target_idx)

                    if next_state not in SPT:
                        heapq.heappush(min_heap, (next_cost, step + 1, next_state, next_step))
        return -1
```
