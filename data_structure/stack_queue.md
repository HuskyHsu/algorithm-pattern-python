# 堆疊與佇列

## 簡介

堆疊的特點是後入先出

![image.png](https://img.fuiboom.com/img/stack.png)

根據這個特點可以臨時保存一些數據，之後用到依次再彈出來，常用於 DFS 深度搜索

佇列一般常用於 BFS 廣度搜索，類似一層一層的搜索

## Stack 堆疊

### [min-stack](https://leetcode.com/problems/min-stack/)

> 設計一個支援 push，pop，top 操作，並能在常數時間內檢索到最小元素的堆疊。

- 思路：用兩個堆疊實現或插入元組實現，保證當前最小值在堆疊頂即可

```Python
class MinStack:

    def __init__(self):
        self.stack = []

    def push(self, x: int) -> None:
        if len(self.stack) > 0:
            self.stack.append((x, min(x, self.stack[-1][1])))
        else:
            self.stack.append((x, x))

    def pop(self) -> int:
        return self.stack.pop()[0]

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]
```

### [evaluate-reverse-polish-notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)

> **波蘭錶達式計算** > **輸入:** ["2", "1", "+", "3", "*"] > **輸出:** 9
> **解釋:** ((2 + 1) \* 3) = 9

- 思路：通過堆疊保存原來的元素，遇到錶達式彈出運算，再推入結果，重複這個過程

```Python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:

        def comp(or1, op, or2):
            if op == '+':
                return or1 + or2

            if op == '-':
                return or1 - or2

            if op == '*':
                return or1 * or2

            if op == '/':
                abs_result = abs(or1) // abs(or2)
                return abs_result if or1 * or2 > 0 else -abs_result

        stack = []

        for token in tokens:
            if token in ['+', '-', '*', '/']:
                or2 = stack.pop()
                or1 = stack.pop()
                stack.append(comp(or1, token, or2))
            else:
                stack.append(int(token))

        return stack[0]
```

### [decode-string](https://leetcode.com/problems/decode-string/)

> 給定一個經過編碼的字符串，返回它解碼後的字符串。<br>
> s = "3[a]2[bc]", 返回 "aaabcbc".<br>
> s = "3[a2[c]]", 返回 "accaccacc".<br>
> s = "2[abc]3[cd]ef", 返回 "abcabccdcdcdef".

- 思路：通過兩個堆疊進行操作，一個用於存數，另一個用來存字符串

```Python
class Solution:
    def decodeString(self, s: str) -> str:

        stack_str = ['']
        stack_num = []

        num = 0
        for c in s:
            if c >= '0' and c <= '9':
                num = num * 10 + int(c)
            elif c == '[':
                stack_num.append(num)
                stack_str.append('')
                num = 0
            elif c == ']':
                cur_str = stack_str.pop()
                stack_str[-1] += cur_str * stack_num.pop()
            else:
                stack_str[-1] += c

        return stack_str[0]
```

### [binary-tree-inorder-traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)

> 給定一個二元樹，返回它的*中序*遍歷。

- [reference](https://en.wikipedia.org/wiki/Tree_traversal#In-order)

```Python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:

        stack, inorder = [], []
        node = root

        while len(stack) > 0 or node is not None:
            if node is not None:
                stack.append(node)
                node = node.left
            else:
                node = stack.pop()
                inorder.append(node.val)
                node = node.right

        return inorder
```

### [clone-graph](https://leetcode.com/problems/clone-graph/)

> 給你無向連通圖中一個節點的引用，請你返回該圖的深拷貝（克隆）。

- BFS

```Python
class Solution:
    def cloneGraph(self, start: 'Node') -> 'Node':

        if start is None:
            return None

        visited = {start: Node(start.val, [])}
        bfs = collections.deque([start])

        while len(bfs) > 0:
            curr = bfs.popleft()
            curr_copy = visited[curr]
            for n in curr.neighbors:
                if n not in visited:
                    visited[n] = Node(n.val, [])
                    bfs.append(n)
                curr_copy.neighbors.append(visited[n])

        return visited[start]
```

- DFS iterative

```Python
class Solution:
    def cloneGraph(self, start: 'Node') -> 'Node':

        if start is None:
            return None

        if not start.neighbors:
            return Node(start.val)

        visited = {start: Node(start.val, [])}
        dfs = [start]

        while len(dfs) > 0:
            peek = dfs[-1]
            peek_copy = visited[peek]
            if len(peek_copy.neighbors) == 0:
                for n in peek.neighbors:
                    if n not in visited:
                        visited[n] = Node(n.val, [])
                        dfs.append(n)
                    peek_copy.neighbors.append(visited[n])
            else:
                dfs.pop()

        return visited[start]
```

### [number-of-islands](https://leetcode.com/problems/number-of-islands/)

> 給定一個由  '1'（陸地）和 '0'（水）組成的的二維網格，計算島嶼的數量。一個島被水包圍，並且它是通過水準方嚮或垂直方嚮上相鄰的陸地連接而成的。你可以假設網格的四個邊均被水包圍。

High-level problem: number of connected component of graph

- 思路：通過深度搜索遍歷可能性（註意標記已訪問元素）

```Python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:

        if not grid or not grid[0]:
            return 0

        m, n = len(grid), len(grid[0])

        def dfs_iter(i, j):
            dfs = []
            dfs.append((i, j))
            while len(dfs) > 0:
                i, j = dfs.pop()
                if grid[i][j] == '1':
                    grid[i][j] = '0'
                    if i - 1 >= 0:
                        dfs.append((i - 1, j))
                    if j - 1 >= 0:
                        dfs.append((i, j - 1))
                    if i + 1 < m:
                        dfs.append((i + 1, j))
                    if j + 1 < n:
                        dfs.append((i, j + 1))
            return

        num_island = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    num_island += 1
                    dfs_iter(i, j)

        return num_island
```

### [largest-rectangle-in-histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)

> 給定 _n_ 個非負整數，用來錶示柱狀圖中各個柱子的高度。每個柱子彼此相鄰，且寬度為 1 。
> 求在該柱狀圖中，能夠勾勒出來的矩形的最大麵積。

- 思路 1：蠻力法，比較每個以 i 開始 j 結束的最大矩形，A(i, j) = (j - i + 1) \* min_height(i, j)，時間複雜度 O(n^2) 無法 AC。

```Python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:

        max_area = 0

        n = len(heights)
        for i in range(n):
            min_height = heights[i]
            for j in range(i, n):
                min_height = min(min_height, heights[j])
                max_area = max(max_area, min_height * (j - i + 1))

        return max_area
```

- 思路 2: 設 A(i, j) 為區間 [i, j) 內最大矩形的麵積，k 為 [i, j) 內最矮 bar 的坐標，則 A(i, j) = max((j - i) \* heights[k], A(i, k), A(k+1, j)), 使用分治法進行求解。時間複雜度 O(nlogn)，其中使用簡單遍歷求最小值無法 AC (最壞情況退化到 O(n^2))，使用線段樹優化後勉強 AC。

```Python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:

        n = len(heights)

        seg_tree = [None] * n
        seg_tree.extend(list(zip(heights, range(n))))
        for i in range(n - 1, 0, -1):
            seg_tree[i] = min(seg_tree[2 * i], seg_tree[2 * i + 1], key=lambda x: x[0])

        def _min(i, j):
            min_ = (heights[i], i)
            i += n
            j += n
            while i < j:
                if i % 2 == 1:
                    min_ = min(min_, seg_tree[i], key=lambda x: x[0])
                    i += 1
                if j % 2 == 1:
                    j -= 1
                    min_ = min(min_, seg_tree[j], key=lambda x: x[0])
                i //= 2
                j //= 2

            return min_

        def LRA(i, j):
            if i == j:
                return 0
            min_k, k = _min(i, j)
            return max(min_k * (j - i), LRA(k + 1, j), LRA(i, k))

        return LRA(0, n)
```

- 思路 3：包含當前 bar 最大矩形的邊界為左邊第一個高度小於當前高度的 bar 和右邊第一個高度小於當前高度的 bar。

```Python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:

        n = len(heights)

        stack = [-1]
        max_area = 0

        for i in range(n):
            while len(stack) > 1 and heights[stack[-1]] > heights[i]:
                h = stack.pop()
                max_area = max(max_area, heights[h] * (i - stack[-1] - 1))
            stack.append(i)

        while len(stack) > 1:
            h = stack.pop()
            max_area = max(max_area, heights[h] * (n - stack[-1] - 1))

        return max_area
```

## Queue 佇列

常用於 BFS 寬度優先搜索

### [implement-queue-using-stacks](https://leetcode.com/problems/implement-queue-using-stacks/)

> 使用堆疊實現佇列

```Python
class MyQueue:

    def __init__(self):
        self.cache = []
        self.out = []

    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.cache.append(x)

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if len(self.out) == 0:
            while len(self.cache) > 0:
                self.out.append(self.cache.pop())

        return self.out.pop()

    def peek(self) -> int:
        """
        Get the front element.
        """
        if len(self.out) > 0:
            return self.out[-1]
        else:
            return self.cache[0]

    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return len(self.cache) == 0 and len(self.out) == 0
```

### [binary-tree-level-order-traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)

> 二元樹的層序遍歷

```Python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:

        levels = []
        if root is None:
            return levels

        bfs = collections.deque([root])

        while len(bfs) > 0:
            levels.append([])

            level_size = len(bfs)
            for _ in range(level_size):
                node = bfs.popleft()
                levels[-1].append(node.val)

                if node.left is not None:
                    bfs.append(node.left)
                if node.right is not None:
                    bfs.append(node.right)

        return levels
```

### [01-matrix](https://leetcode.com/problems/01-matrix/)

> 給定一個由 0 和 1 組成的矩陣，找出每個元素到最近的 0 的距離。
> 兩個相鄰元素間的距離為 1

- 思路 1: 從 0 開始 BFS, 遇到距離最小值需要更新的則更新後重新入隊更新後續結點

```Python
class Solution:
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:

        if len(matrix) == 0 or len(matrix[0]) == 0:
            return matrix

        m, n = len(matrix), len(matrix[0])
        dist = [[float('inf')] * n for _ in range(m)]

        bfs = collections.deque([])
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    dist[i][j] = 0
                    bfs.append((i, j))

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while len(bfs) > 0:
            i, j = bfs.popleft()
            for dn_i, dn_j in neighbors:
                n_i, n_j = i + dn_i, j + dn_j
                if n_i >= 0 and n_i < m and n_j >= 0 and n_j < n:
                    if dist[n_i][n_j] > dist[i][j] + 1:
                        dist[n_i][n_j] = dist[i][j] + 1
                        bfs.append((n_i, n_j))

        return dist
```

- 思路 2: 2-pass DP，dist(i, j) = max{dist(i - 1, j), dist(i + 1, j), dist(i, j - 1), dist(i, j + 1)} + 1

```Python
class Solution:
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:

        if len(matrix) == 0 or len(matrix[0]) == 0:
            return matrix

        m, n = len(matrix), len(matrix[0])

        dist = [[float('inf')] * n for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 1:
                    if i - 1 >= 0:
                        dist[i][j] = min(dist[i - 1][j] + 1, dist[i][j])
                    if j - 1 >= 0:
                        dist[i][j] = min(dist[i][j - 1] + 1, dist[i][j])
                else:
                    dist[i][j] = 0

        for i in range(-1, -m - 1, -1):
            for j in range(-1, -n - 1, -1):
                if matrix[i][j] == 1:
                    if i + 1 < 0:
                        dist[i][j] = min(dist[i + 1][j] + 1, dist[i][j])
                    if j + 1 < 0:
                        dist[i][j] = min(dist[i][j + 1] + 1, dist[i][j])

        return dist
```

## 補充：單調堆疊

顧名思義，單調堆疊即是堆疊中元素有單調性的堆疊，典型應用為用線性的時間複雜度找左右兩側第一個大於/小於當前元素的位置。

### [largest-rectangle-in-histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)

```Python
class Solution:
    def largestRectangleArea(self, heights) -> int:
        heights.append(0)
        stack = [-1]
        result = 0
        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                cur = stack.pop()
                result = max(result, heights[cur] * (i - stack[-1] - 1))
            stack.append(i)
        return result
```

### [trapping-rain-water](https://leetcode.com/problems/trapping-rain-water/)

```Python
class Solution:
    def trap(self, height: List[int]) -> int:

        stack = []
        result = 0

        for i in range(len(height)):
            while stack and height[i] > height[stack[-1]]:
                cur = stack.pop()
                if not stack:
                    break
                result += (min(height[stack[-1]], height[i]) - height[cur]) * (i - stack[-1] - 1)
            stack.append(i)

        return result
```

## 補充：單調佇列

單調堆疊的拓展，可以從數組頭 pop 出舊元素，典型應用是以線性時間獲得區間最大/最小值。

### [sliding-window-maximum](https://leetcode.com/problems/sliding-window-maximum/)

> 求滑動視窗中的最大元素

```Python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:

        N = len(nums)
        if N * k == 0:
            return []

        if k == 1:
            return nums[:]

        # define a max queue
        maxQ = collections.deque()

        result = []
        for i in range(N):
            if maxQ and maxQ[0] == i - k:
                maxQ.popleft()

            while maxQ and nums[maxQ[-1]] < nums[i]:
                maxQ.pop()

            maxQ.append(i)

            if i >= k - 1:
                result.append(nums[maxQ[0]])

        return result
```

### [shortest-subarray-with-sum-at-least-k](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/)

```Python
class Solution:
    def shortestSubarray(self, A: List[int], K: int) -> int:
        N = len(A)
        cdf = [0]
        for num in A:
            cdf.append(cdf[-1] + num)

        result = N + 1
        minQ = collections.deque()

        for i, csum in enumerate(cdf):

            while minQ and csum <= cdf[minQ[-1]]:
                minQ.pop()

            while minQ and csum - cdf[minQ[0]] >= K:
                result = min(result, i - minQ.popleft())

            minQ.append(i)

        return result if result < N + 1 else -1
```

## 總結

- 熟悉堆疊的使用場景
  - 後入先出，保存臨時值
  - 利用堆疊 DFS 深度搜索
- 熟悉佇列的使用場景
  - 利用佇列 BFS 廣度搜索

## 練習

- [ ] [min-stack](https://leetcode.com/problems/min-stack/)
- [ ] [evaluate-reverse-polish-notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)
- [ ] [decode-string](https://leetcode.com/problems/decode-string/)
- [ ] [binary-tree-inorder-traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)
- [ ] [clone-graph](https://leetcode.com/problems/clone-graph/)
- [ ] [number-of-islands](https://leetcode.com/problems/number-of-islands/)
- [ ] [largest-rectangle-in-histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
- [ ] [implement-queue-using-stacks](https://leetcode.com/problems/implement-queue-using-stacks/)
- [ ] [01-matrix](https://leetcode.com/problems/01-matrix/)
