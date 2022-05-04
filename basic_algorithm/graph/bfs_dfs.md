# 深度優先搜索，廣度優先搜索

### 深度優先搜索樣板

- 先序，遞迴

```Python
def DFS(x):
    visit(x)
    for n in neighbor(x):
        if not visited(n):
            DFS(n)
    return
```

- 先序，叠代，出堆疊時訪問

```Python
def DFS(x):
    dfs = [x] # implement by a stack
    while dfs:
        v = dfs.pop()
        if not visited(v):
            visit(v)
            for n in neighbor(v):
                if not visited(n):
                    dfs.append(n)
    return
```

- 後序，遞迴

```Python
def DFS(x): # used when need to aggregate results from children
    discovering(x)
    for n in neighbor(x):
        if not discovering(n) and not visited(n):
            DFS(n)
    visit(x)
    return
```

### 廣度優先搜索樣板

相對於 dfs 可能收斂更慢，但是可以用來找不帶權的最短路徑

- 以結點為單位搜索

```Python
def BFS(x):
    visit(x)
    bfs = collections.deque([x])
    while bfs:
        v = bfs.popleft()
        for n in neighbor(v):
            if not visited(n):
                visit(n)
                bfs.append(n)
    return
```

- 以層為單位搜索，典型應用是找不帶權的最短路徑

```Python
def BFS(x):
    visit(x)
    bfs = collections.deque([x])
    while bfs:
        num_level = len(bfs)
        for _ in range(num_level)
            v = bfs.popleft()
            for n in neighbor(v):
                if not visited(v):
                    visit(n)
                    bfs.append(n)
    return
```

## 例題

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

### [sliding-puzzle](https://leetcode.com/problems/sliding-puzzle)

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

        if start == target:
            return 0

        SPT = set([start])
        bfs = collections.deque([(start, start.index(0))])

        step = 1
        while bfs:
            num_level = len(bfs)
            for _ in range(num_level):
                state, idx0 = bfs.popleft()

                for next_step in next_move[idx0]:
                    next_state = list(state)
                    next_state[idx0], next_state[next_step] = next_state[next_step], next_state[idx0]
                    next_state = tuple(next_state)

                    if next_state == target:
                        return step

                    if next_state not in SPT:
                        SPT.add(next_state)
                        bfs.append((next_state, next_step))
            step += 1
        return -1
```
