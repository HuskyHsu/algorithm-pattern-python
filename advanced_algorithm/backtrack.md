# 回溯法

## 背景

回溯法（backtrack）常用於遍歷列錶所有子集，是 DFS 深度搜索一種，一般用於全排列，窮盡所有可能，遍歷的過程實際上是一個決策樹的遍歷過程。時間複雜度一般 O(N!)，它不像動態規劃存在重疊子問題可以優化，回溯算法就是純暴力窮舉，複雜度一般都很高。

## 樣板

```go
result = []
func backtrack(選擇列錶,路徑):
    if 滿足結束條件:
        result.add(路徑)
        return
    for 選擇 in 選擇列錶:
        做選擇
        backtrack(選擇列錶,路徑)
        撤銷選擇
```

核心就是從選擇列錶裏做一個選擇，然後一直遞迴往下搜索答案，如果遇到路徑不通，就返回來撤銷這次選擇。

## 示例

### [subsets](https://leetcode.com/problems/subsets/)

> 給定一組不含重複元素的整數數組 nums，返回該數組所有可能的子集（冪集）。

遍歷過程

![image.png](https://img.fuiboom.com/img/backtrack.png)

```Python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:

        n = len(nums)
        result = []

        def backtrack(start, k, route=[]):
            if len(route) == k:
                result.append(route.copy())
                return

            for i in range(start, n):
                route.append(nums[i])
                backtrack(i + 1, k)
                route.pop()

            return

        for k in range(n + 1):
            backtrack(0, k)

        return result
```

### [subsets-ii](https://leetcode.com/problems/subsets-ii/)

> 給定一個可能包含重複元素的整數數組 nums，返回該數組所有可能的子集（冪集）。說明：解集不能包含重複的子集。

```Python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:

        nums = sorted(nums)
        n = len(nums)
        result = []

        def backtrack(start, k, route=[]):

            if len(route) == k:
                result.append(route.copy())
                return

            last = None
            for i in range(start, n):
                if nums[i] != last:
                    route.append(nums[i])
                    backtrack(i + 1, k)
                    last = route.pop()

            return

        for k in range(n + 1):
            backtrack(0, k)

        return result
```

### [permutations](https://leetcode.com/problems/permutations/)

> 給定一個冇有重複數字的序列，返回其所有可能的全排列。

- 思路 1：需要記錄已經選擇過的元素，滿足條件的結果才進行返回，需要額外 O(n) 的空間

```Python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:

        n = len(nums)
        result = []

        in_route = [False] * n

        def backtrack(route=[]):

            if len(route) == n:
                result.append(route.copy())
                return

            for i in range(n):
                if not in_route[i]:
                    route.append(nums[i])
                    in_route[i] = True
                    backtrack()
                    route.pop()
                    in_route[i] = False

            return

        backtrack()
        return result
```

- 思路 2: 針對此題的更高級的回溯，利用原有的數組，每次回溯將新選擇的元素與當前位置元素交換，回溯完成再換回來

```Python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:

        n = len(nums)
        result = []

        def backtrack(idx=0):
            if idx == n:
                result.append(nums.copy())
            for i in range(idx, n):
                nums[idx], nums[i] = nums[i], nums[idx]
                backtrack(idx + 1)
                nums[idx], nums[i] = nums[i], nums[idx]
            return

        backtrack()
        return result
```

### [permutations-ii](https://leetcode.com/problems/permutations-ii/)

> 給定一個可包含重複數字的序列，返回所有不重複的全排列。

註意此題（貌似）無法使用上題的思路 2，因為交換操作會打亂排序。

```Python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:

        nums = sorted(nums)
        n = len(nums)
        result = []

        in_route = [False] * n

        def backtrack(route=[]):

            if len(route) == n:
                result.append(route.copy())
                return

            last = None
            for i in range(n):
                if not in_route[i] and nums[i] != last:
                    route.append(nums[i])
                    in_route[i] = True
                    backtrack()
                    last = route.pop()
                    in_route[i] = False

            return

        backtrack()
        return result
```

### [combination-sum](https://leetcode.com/problems/combination-sum/)

```Python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:

        n = len(candidates)
        result = []

        def backtrack(first=0, route=[], route_sum=0):

            if route_sum == target:
                result.append(route.copy())
                return

            if route_sum > target:
                return

            for i in range(first, n):
                route.append(candidates[i])
                route_sum += candidates[i]
                backtrack(i, route, route_sum)
                route_sum -= route.pop()

            return

        backtrack()
        return result
```

### [letter-combinations-of-a-phone-number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)

```Python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:

        n = len(digits)
        result = []

        if n == 0:
            return result

        num2char = {
            '2': ['a', 'b', 'c'],
            '3': ['d', 'e', 'f'],
            '4': ['g', 'h', 'i'],
            '5': ['j', 'k', 'l'],
            '6': ['m', 'n', 'o'],
            '7': ['p', 'q', 'r', 's'],
            '8': ['t', 'u', 'v'],
            '9': ['w', 'x', 'y', 'z']
        }

        def backtrack(idx=0, route=[]):
            if idx == n:
                result.append(''.join(route))
                return

            for c in num2char[digits[idx]]:
                route.append(c)
                backtrack(idx + 1, route)
                route.pop()

            return

        backtrack()
        return result
```

### [palindrome-partitioning](https://leetcode.com/problems/palindrome-partitioning/)

```Python
class Solution:
    def partition(self, s: str) -> List[List[str]]:

        N = len(s)
        Pal = collections.defaultdict(set)

        def isPal(i, j):
            if i < j:
                return j in Pal[i]
            return True

        for j in range(N):
            for i in range(j + 1):
                if s[i] == s[j] and isPal(i + 1, j - 1):
                    Pal[i].add(j)

        result = []

        def backtrack(first=0, route=[]):

            if first == N:
                result.append(route[:])
                return

            for i in Pal[first]:
                route.append(s[first:i+1])
                backtrack(i + 1)
                route.pop()

            return

        backtrack()
        return result
```

### [restore-ip-addresses](https://leetcode.com/problems/restore-ip-addresses/)

```Python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:

        n = len(s)
        result = []

        if n > 12:
            return result

        def Valid_s(i, j):
            return i < j and j <= n and ((s[i] != '0' and int(s[i:j]) < 256) or (s[i] == '0' and i == j - 1))

        def backtrack(start=0, route=[]):

            if len(route) == 3:
                if Valid_s(start, n):
                    result.append('.'.join(route) + '.' + s[start:])
                return

            for i in range(start, start + 3):
                if Valid_s(start, i + 1):
                    route.append(s[start:i + 1])
                    backtrack(i + 1, route)
                    route.pop()

            return

        backtrack()
        return result
```

## 練習

- [ ] [subsets](https://leetcode.com/problems/subsets/)
- [ ] [subsets-ii](https://leetcode.com/problems/subsets-ii/)
- [ ] [permutations](https://leetcode.com/problems/permutations/)
- [ ] [permutations-ii](https://leetcode.com/problems/permutations-ii/)

- [ ] [combination-sum](https://leetcode.com/problems/combination-sum/)
- [ ] [letter-combinations-of-a-phone-number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)
- [ ] [palindrome-partitioning](https://leetcode.com/problems/palindrome-partitioning/)
- [ ] [restore-ip-addresses](https://leetcode.com/problems/restore-ip-addresses/)
