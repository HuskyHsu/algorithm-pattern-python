# 拓撲排序

圖的拓撲排序 (topological sorting) 一般用於給定一係列偏序關係，求一個全序關係的題目中。以元素為結點，以偏序關係為邊構造有嚮圖，然後應用拓撲排序算法即可得到全序關係。

### [course-schedule-ii](https://leetcode.com/problems/course-schedule-ii/)

> 給定課程的先修關係，求一個可行的修課順序

非常經典的拓撲排序應用。下麵給出 3 種實現方法，可以當做模闆使用。

- 方法 1：DFS 的遞迴實現

```Python
NOT_VISITED = 0
DISCOVERING = 1
VISITED = 2

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:

        # construct graph
        graph_neighbor = collections.defaultdict(list)
        for course, pre in prerequisites:
            graph_neighbor[pre].append(course)

        # recursive postorder DFS for topological sort
        tsort_rev = []
        status = [NOT_VISITED] * numCourses

        def dfs(course):
            status[course] = DISCOVERING
            for n in graph_neighbor[course]:
                if status[n] == DISCOVERING or (status[n] == NOT_VISITED and not dfs(n)):
                    return False
            tsort_rev.append(course)
            status[course] = VISITED
            return True

        for course in range(numCourses):
            if status[course] == NOT_VISITED and not dfs(course):
                return []

        return tsort_rev[::-1]
```

- 方法 2：DFS 的叠代實現

```Python
NOT_VISITED = 0
DISCOVERING = 1
VISITED = 2

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:

        # construct graph
        graph_neighbor = collections.defaultdict(list)
        for course, pre in prerequisites:
            graph_neighbor[pre].append(course)

        # iterative postorder DFS for topological sort
        tsort_rev = []
        status = [NOT_VISITED] * numCourses

        dfs = []
        for course in range(numCourses):
            if status[course] == NOT_VISITED:
                dfs.append(course)
                status[course] = DISCOVERING

                while dfs:
                    if graph_neighbor[dfs[-1]]:
                        n = graph_neighbor[dfs[-1]].pop()
                        if status[n] == DISCOVERING:
                            return []
                        if status[n] == NOT_VISITED:
                            dfs.append(n)
                            status[n] = DISCOVERING
                    else:
                        tsort_rev.append(dfs.pop())
                        status[tsort_rev[-1]] = VISITED

        return tsort_rev[::-1]
```

- 方法 3：[Kahn's algorithm](https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm)

```Python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:

        # construct graph with indegree data
        graph_neighbor = collections.defaultdict(list)
        indegree = collections.defaultdict(int)

        for course, pre in prerequisites:
            graph_neighbor[pre].append(course)
            indegree[course] += 1

        # Kahn's algorithm
        src_cache = [] # can also use queue
        for i in range(numCourses):
            if indegree[i] == 0:
                src_cache.append(i)

        tsort = []
        while src_cache:
            tsort.append(src_cache.pop())
            for n in graph_neighbor[tsort[-1]]:
                indegree[n] -= 1
                if indegree[n] == 0:
                    src_cache.append(n)

        return tsort if len(tsort) == numCourses else []
```

### [alien-dictionary](https://leetcode.com/problems/alien-dictionary/)

```Python
class Solution:
    def alienOrder(self, words: List[str]) -> str:

        N = len(words)

        if N == 0:
            return ''

        if N == 1:
            return words[0]

        # construct graph
        indegree = {c: 0 for word in words for c in word}
        graph = collections.defaultdict(list)

        for i in range(N - 1):
            first, second = words[i], words[i + 1]
            len_f, len_s = len(first), len(second)
            find_different = False
            for j in range(min(len_f, len_s)):
                f, s = first[j], second[j]
                if f != s:
                    if s not in graph[f]:
                        graph[f].append(s)
                        indegree[s] += 1
                    find_different = True
                    break

            if not find_different and len_f > len_s:
                return ''

        tsort = []
        src_cache = [c for c in indegree if indegree[c] == 0]

        while src_cache:
            tsort.append(src_cache.pop())
            for n in graph[tsort[-1]]:
                indegree[n] -= 1
                if indegree[n] == 0:
                    src_cache.append(n)

        return ''.join(tsort) if len(tsort) == len(indegree) else ''
```

### [sequence-reconstruction](https://leetcode.com/problems/sequence-reconstruction/)

Kahn's algorithm 可以判斷拓撲排序是否唯一。

```Python
class Solution:
    def sequenceReconstruction(self, org: List[int], seqs: List[List[int]]) -> bool:

        N = len(org)
        inGraph = [False] * (N + 1)
        graph_set = collections.defaultdict(set)
        for seq in seqs:
            if seq:
                if seq[0] > N or seq[0] < 1:
                    return False
                inGraph[seq[0]] = True
                for i in range(1, len(seq)):
                    if seq[i] > N or seq[i] < 1:
                        return False
                    inGraph[seq[i]] = True
                    graph_set[seq[i - 1]].add(seq[i])

        indegree = collections.defaultdict(int)
        for node in graph_set:
            for n in graph_set[node]:
                indegree[n] += 1

        num_valid, count0, src = 0, -1, 0
        for i in range(1, N + 1):
            if inGraph[i] and indegree[i] == 0:
                count0 += 1
                src = i

        i = 0
        while count0 == i and src == org[i]:
            num_valid += 1
            for n in graph_set[src]:
                indegree[n] -= 1
                if indegree[n] == 0:
                    count0 += 1
                    src = n
            i += 1

        return num_valid == N
```
