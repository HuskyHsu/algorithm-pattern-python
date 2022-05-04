# 堆積

用到優先佇列 (priority queue) 或堆積 (heap) 的題一般需要維護一個動態更新的池，元素會被頻繁加入到池中或從池中被取走，每次取走的元素為池中優先級最高的元素 (可以簡單理解為最大或者最小)。用堆來實現優先級隊列是效率非常高的方法，加入或取出都隻需要 O(log N) 的複雜度。

## Kth largest/smallest

找數據中第 K 個最大/最小數據是堆的一個典型應用。以找最大為例，遍曆數據時，使用一個最小堆來維護當前最大的 K 個數據，堆頂數據為這 K 個數據中最小，即是你想要的第 k 個最大數據。每檢查一個新數據，判斷是否大於堆頂，若大於，說明堆頂數據小於了 K 個值，不是我們想找的第 K 個最大，則將新數據 push 進堆並 pop 掉堆頂，若小於則不操作，這樣當遍曆完全部數據後堆頂即為想要的結果。找最小時換成最大堆即可。

### [kth-largest-element-in-a-stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/)

> 設計一個找到數據流中第 K 大元素的類。

```Python
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.K = k
        self.min_heap = []
        for num in nums:
            if len(self.min_heap) < self.K:
                heapq.heappush(self.min_heap, num)
            elif num > self.min_heap[0]:
                heapq.heappushpop(self.min_heap, num)

    def add(self, val: int) -> int:
        if len(self.min_heap) < self.K:
            heapq.heappush(self.min_heap, val)
        elif val > self.min_heap[0]:
            heapq.heappushpop(self.min_heap, val)

        return self.min_heap[0]
```

### [kth-smallest-element-in-a-sorted-matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)

> 給定一個 n x n 矩陣，其中每行和每列元素均按升序排序，找到矩陣中第 k 小的元素。

- 此題使用 heap 來做並不是最優做法，相當於 N 個 sorted list 裏找第 k 個最小，列有序的條件冇有充分利用，但是卻是比較容易想且比較通用的做法。

```Python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:

        N = len(matrix)

        min_heap = []
        for i in range(min(k, N)): # 這裏用了一點列有序的性質，第k個最小隻可能在前k行中(k行以後的數至少大於了k個數)
            min_heap.append((matrix[i][0], i, 0))

        heapq.heapify(min_heap)

        while k > 0:
            num, r, c = heapq.heappop(min_heap)

            if c < N - 1:
                heapq.heappush(min_heap, (matrix[r][c + 1], r, c + 1))

            k -= 1

        return num
```

### [find-k-pairs-with-smallest-sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/)

```Python
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:

        m, n = len(nums1), len(nums2)
        result = []

        if m * n == 0:
            return result

        min_heap = [(nums1[0] + nums2[0], 0, 0)]
        seen = set()

        while min_heap and len(result) < k:
            _, i1, i2 = heapq.heappop(min_heap)
            result.append([nums1[i1], nums2[i2]])
            if i1 < m - 1 and (i1 + 1, i2) not in seen:
                heapq.heappush(min_heap, (nums1[i1 + 1] + nums2[i2], i1 + 1, i2))
                seen.add((i1 + 1, i2))
            if i2 < n - 1 and (i1, i2 + 1) not in seen:
                heapq.heappush(min_heap, (nums1[i1] + nums2[i2 + 1], i1, i2 + 1))
                seen.add((i1, i2 + 1))

        return result
```

## Greedy + Heap

Heap 可以高效地取出或更新當前池中優先級最高的元素，因此適用於一些需要 greedy 算法的場景。

### [maximum-performance-of-a-team](https://leetcode.com/problems/maximum-performance-of-a-team/)

> 公司有 n 個專案師，給兩個數組 speed 和 efficiency，其中 speed[i] 和 efficiency[i] 分別代錶第 i 位專案師的速度和效率。請你返回由最多 k 個專案師組成的團隊的最大錶現值。錶現值的定義為：一個團隊中所有專案師速度的和乘以他們效率值中的最小值。

- [See my review here.](<https://leetcode.com/problems/maximum-performance-of-a-team/discuss/741822/Met-this-problem-in-my-interview!!!-(Python3-greedy-with-heap)>) [或者這裏(中文)](https://leetcode.com/problems/maximum-performance-of-a-team/solution/greedy-with-min-heap-lai-zi-zhen-shi-mian-shi-de-j/)

```Python
# similar question: LC 857
class Solution:
    def maxPerformance(self, n, speed, efficiency, k):

        people = sorted(zip(speed, efficiency), key=lambda x: -x[1])

        result, sum_speed = 0, 0
        min_heap = []

        for i, (s, e) in enumerate(people):
            if i < k:
                sum_speed += s
                heapq.heappush(min_heap, s)
            elif s > min_heap[0]:
                sum_speed += s - heapq.heappushpop(min_heap, s)

            result = max(result, sum_speed * e)

        return result #% 1000000007
```

### [ipo](https://leetcode.com/problems/ipo/)

- 貪心策略為每次做當前成本範圍內利潤最大的項目。

```Python
class Solution:
    def findMaximizedCapital(self, k: int, W: int, Profits: List[int], Capital: List[int]) -> int:
        N = len(Profits)
        projects = sorted([(-Profits[i], Capital[i]) for i in range(N)], key=lambda x: x[1])

        projects.append((0, float('inf')))

        max_profit_heap = []

        for i in range(N + 1):
            while projects[i][1] > W and len(max_profit_heap) > 0 and k > 0:
                W -= heapq.heappop(max_profit_heap)
                k -= 1

            if projects[i][1] > W or k == 0:
                break

            heapq.heappush(max_profit_heap, projects[i][0])

        return W
```

### [meeting-rooms-ii](https://leetcode.com/problems/meeting-rooms-ii/)

- 此題用 greedy + heap 解並不是很 intuitive，存在複雜度相同但更簡單直觀的做法。

```Python
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:

        if len(intervals) == 0: return 0

        intervals.sort(key=lambda item: item[0])
        end_times = [intervals[0][1]]

        for interval in intervals[1:]:
            if end_times[0] <= interval[0]:
                heapq.heappop(end_times)

            heapq.heappush(end_times, interval[1])

        return len(end_times)
```

### [reorganize-string](https://leetcode.com/problems/reorganize-string/)

> 給定一個字符串 S，檢查是否能重新排佈其中的字母，使得任意兩相鄰的字符不同。若可行，輸出任意可行的結果。若不可行，返回空字符串。

- 貪心策略為每次取前兩個最多數量的字母加入到結果。

```Python
class Solution:
    def reorganizeString(self, S: str) -> str:

        max_dup = (len(S) + 1) // 2
        counts = collections.Counter(S)

        heap = []
        for c, f in counts.items():
            if f > max_dup:
                return ''
            heap.append([-f, c])
        heapq.heapify(heap)

        result = []
        while len(heap) > 1:
            first = heapq.heappop(heap)
            result.append(first[1])
            first[0] += 1
            second = heapq.heappop(heap)
            result.append(second[1])
            second[0] += 1

            if first[0] < 0:
                heapq.heappush(heap, first)
            if second[0] < 0:
                heapq.heappush(heap, second)

        if len(heap) == 1:
            result.append(heap[0][1])

        return ''.join(result)
```

### Prim's Algorithm

實現上是 greedy + heap 的一個應用，用於構造圖的最小生成樹 (MST)。

### [minimum-risk-path](https://www.lintcode.com/problem/minimum-risk-path/description)

> 地圖上有 m 條無嚮邊，每條邊 (x, y, w) 錶示位置 m 到位置 y 的權值為 w。從位置 0 到 位置 n 可能有多條路徑。我們定義一條路徑的危險值為這條路徑中所有的邊的最大權值。請問從位置 0 到 位置 n 所有路徑中最小的危險值為多少？

- 最小危險值為最小生成樹中 0 到 n 路徑上的最大邊權。

```Python
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

### Dijkstra's Algorithm

實現上是 greedy + heap 的一個應用，用於求解圖的單源最短路徑相關的問題，生成的樹為最短路徑樹 (SPT)。

### [network-delay-time](https://leetcode.com/problems/network-delay-time/)

- 標準的單源最短路徑問題，使用樸素的的 Dijikstra 算法即可，可以當成模闆使用。

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
