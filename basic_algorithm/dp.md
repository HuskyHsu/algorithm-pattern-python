# 動態規劃

## 背景

先從一道題目開始~

如題  [triangle](https://leetcode.com/problems/triangle/)

> 給定一個三角形，找出自頂嚮下的最小路徑和。每一步隻能移動到下一行中相鄰的結點上。

例如，給定三角形：

```text
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
```

自頂嚮下的最小路徑和為  11（即，2 + 3 + 5 + 1 = 11）。

使用 DFS（遍歷 或者 分治法）

遍歷

![image.png](https://img.fuiboom.com/img/dp_triangle.png)

分治法

![image.png](https://img.fuiboom.com/img/dp_dc.png)

優化 DFS，緩存已經被計算的值（稱為：記憶化搜索 本質上：動態規劃）

![image.png](https://img.fuiboom.com/img/dp_memory_search.png)

動態規劃就是把大問題變成小問題，並解決了小問題重複計算的方法稱為動態規劃

動態規劃和 DFS 區別

- 二元樹 子問題是有有交集，所以大部分二元樹都用遞迴或者分治法，即 DFS，就可以解決
- 像 triangle 這種是有重複走的情況，**子問題是有交集**，所以可以用動態規劃來解決

動態規劃，自底嚮上

```Python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if len(triangle) == 0:
            return 0

        dp = triangle[-1].copy()

        for i in range(-2, -len(triangle) - 1, -1):
            for j in range(len(triangle[i])):
                dp[j] = triangle[i][j] + min(dp[j], dp[j + 1])

        return dp[0]

```

動態規劃，自頂嚮下

```Python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if len(triangle) == 0:
            return 0

        dp = triangle[0]
        for row in triangle[1:]:
            dp_new = [row[0] + dp[0]]
            for i in range(len(dp) - 1):
                dp_new.append(row[i+1] + min(dp[i], dp[i+1]))
            dp_new.append(row[-1] + dp[-1])
            dp = dp_new

        return min(dp)
```

## 遞迴和動規關係

遞迴是一種程式的實現方式：函數的自我調用

```go
Function(x) {
	...
	Funciton(x-1);
	...
}
```

動態規劃：是一種解決問題的思想，大規模問題的結果，是由小規模問題的結果運算得來的。動態規劃可用遞迴來實現(Memorization Search)

## 使用場景

滿足兩個條件

- 滿足以下條件之一
  - 求最大/最小值（Maximum/Minimum ）
  - 求是否可行（Yes/No ）
  - 求可行個數（Count(\*) ）
- 滿足不能排序或者交換（Can not sort / swap ）

如題：[longest-consecutive-sequence](https://leetcode.com/problems/longest-consecutive-sequence/)  位置可以交換，所以不用動態規劃

## 四點要素

1. **狀態 State**
   - 靈感，創造力，存儲小規模問題的結果
2. 方程 Function
   - 狀態之間的聯係，怎麼通過小的狀態，來算大的狀態
3. 初始化 Intialization
   - 最極限的小狀態是什麼, 起點
4. 答案 Answer
   - 最大的那個狀態是什麼，終點

## 常見四種類型

1. Matrix DP (10%)
1. Sequence (40%)
1. Two Sequences DP (40%)
1. Backpack (10%)

> 注意點
>
> - 貪心算法大多題目靠背答案，所以如果能用動態規劃就盡量用動規，不用貪心算法

## 1、矩陣類型（10%）

### [minimum-path-sum](https://leetcode.com/problems/minimum-path-sum/)

> 給定一個包含非負整數的  *m* x *n*  網格，請找出一條從左上角到右下角的路徑，使得路徑上的數字總和為最小。

思路：動態規劃

1. state: f(x, y) 從起點走到 (x, y) 的最短路徑

2. function: f(x, y) = min(f(x - 1, y), f(x, y - 1]) + A(x, y)

3. intialize: f(0, 0) = A(0, 0)、f(i, 0) = sum(0,0 -> i,0)、 f(0, i) = sum(0,0 -> 0,i)

4. answer: f(n - 1, m - 1)

5. 2D DP -> 1D DP

```Python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        dp = [0] * n
        dp[0] = grid[0][0]
        for i in range(1, n):
            dp[i] = dp[i-1] + grid[0][i]

        for i in range(1, m):
            dp[0] += grid[i][0]
            for j in range(1, n):
                dp[j] = grid[i][j] + min(dp[j-1], dp[j])
        return dp[-1]
```

### [unique-paths](https://leetcode.com/problems/unique-paths/)

> 一個機器人位於一個 m x n 網格的左上角 （起始點在下圖中標記為“Start” ）。
> 機器人每次隻能嚮下或者嚮右移動一步。機器人試圖達到網格的右下角（在下圖中標記為“Finish”）。
> 問總共有多少條不同的路徑？

```Python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:

        if m < n:
            m, n = n, m

        dp = [1] * n

        for i in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]

        return dp[-1]
```

### [unique-paths-ii](https://leetcode.com/problems/unique-paths-ii/)

> 一個機器人位於一個 m x n 網格的左上角 （起始點在下圖中標記為“Start” ）。
> 機器人每次隻能嚮下或者嚮右移動一步。機器人試圖達到網格的右下角（在下圖中標記為“Finish”）。
> 問總共有多少條不同的路徑？
> 現在考慮網格中有障礙物。那麼從左上角到右下角將會有多少條不同的路徑？

```Python
class Solution:
    def uniquePathsWithObstacles(self, G: List[List[int]]) -> int:

        m, n = len(G), len(G[0])

        dp = [1] if G[0][0] == 0 else [0]
        for i in range(1, n):
            new = dp[i-1] if G[0][i] == 0 else 0
            dp.append(new)

        for i in range(1, m):
            dp[0] = 0 if G[i][0] == 1 else dp[0]
            for j in range(1, n):
                dp[j] = dp[j-1] + dp[j] if G[i][j] == 0 else 0

        return dp[-1]
```

## 2、序列類型（40%）

### [climbing-stairs](https://leetcode.com/problems/climbing-stairs/)

> 假設你正在爬樓梯。需要  *n*  階你才能到達樓頂。

```Python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n < 2: return n

        step1, step2 = 2, 1

        for _ in range(n - 2):
            step1, step2 = step1 + step2, step1

        return step1
```

### [jump-game](https://leetcode.com/problems/jump-game/)

> 給定一個非負整數數組，你最初位於數組的第一個位置。
> 數組中的每個元素代錶你在該位置可以跳躍的最大長度。
> 判斷你是否能夠到達最後一個位置。

解法：直接 DP 無法得到 O(n)的解，考慮間接 DP

- tail to head

```Python
class Solution:
    def canJump(self, nums: List[int]) -> bool:

        left = len(nums) - 1 # most left index that can reach the last index

        for i in range(len(nums) - 2, -1, -1):

            left = i if i + nums[i] >= left else left # DP

        return left == 0
```

- head to tail

```Python
class Solution:
    def canJump(self, nums: List[int]) -> bool:

        max_pos = nums[0] # furthest index can reach

        for i in range(1, len(nums)):
            if max_pos < i:
                return False
            max_pos = max(max_pos, i + nums[i]) # DP

        return True
```

### [jump-game-ii](https://leetcode.com/problems/jump-game-ii/)

> 給定一個非負整數數組，你最初位於數組的第一個位置。
> 數組中的每個元素代錶你在該位置可以跳躍的最大長度。
> 你的目標是使用最少的跳躍次數到達數組的最後一個位置。

```Python
class Solution:
    def jump(self, nums: List[int]) -> int:

        cur_max = 0
        step_max = 0
        step = 0

        for i in range(len(nums)):

            if cur_max < i: # can't reach i, don't have to consider in this problem
                return float('inf')

            if step_max < i: # can't reach i in current number of steps
                step += 1
                step_max = cur_max

            cur_max = max(cur_max, i + nums[i]) # DP

        return min_step
```

### [palindrome-partitioning-ii](https://leetcode.com/problems/palindrome-partitioning-ii/)

> 給定一個字符串 _s_，將 _s_ 分割成一些子串，使每個子串都是回文串。
> 返回符合要求的最少分割次數。

- Why is hard

僅目標 DP, 判斷回文時間複雜度高 -> 目標 DP + 回文二維 DP, 回文 DP 空間複雜度高 -> 一點 trick, 回文 DP 空間複雜度降為線性

```Python
class Solution:

    def minCut(self, s: str) -> int:

        dp_min = [0] * len(s)
        dp_pal = [True] * len(s)

        def isPal(i, j):
            dp_pal[i] = (s[i] == s[j] and dp_pal[i+1])
            return dp_pal[i]

        for j in range(1, len(s)):

            min_cut = dp_min[j - 1] + 1

            if isPal(0, j):
                min_cut = 0

            for i in range(1, j):
                if isPal(i, j):
                    min_cut = min(min_cut, dp_min[i - 1] + 1)

            dp_min[j] = min_cut

        return dp_min[-1]
```

### [longest-increasing-subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)

> 給定一個無序的整數數組，找到其中最長上升子序列的長度。

- DP(i) 等於以第 i 個數結尾的最長上升子序列的長度，容易想但不是最優

```Python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:

        if len(nums) == 0: return 0

        dp_max = [1] * len(nums)

        for j in range(1, len(nums)):
            for i in range(j):
                if nums[j] > nums[i]:
                    dp_max[j] = max(dp_max[j], dp_max[i] + 1)

        return max(dp_max)
```

- 最優算法使用 greedy + binary search，比較 tricky

```Python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:

        if len(nums) == 0: return 0

        seq = [nums[0]]

        for i in range(1, len(nums)):
            ins = bisect.bisect_left(seq, nums[i])
            if ins == len(seq):
                seq.append(nums[i])
            else:
                seq[ins] = nums[i]

        return len(seq)
```

### [word-break](https://leetcode.com/problems/word-break/)

> 給定一個**非空**字符串  *s*  和一個包含**非空**單詞列錶的字典  *wordDict*，判定  *s*  是否可以被空格拆分為一個或多個在字典中出現的單詞。

```Python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:

        dp = [False] * (len(s) + 1)
        dp[-1] = True

        for j in range(len(s)):
            for i in range(j+1):
                if dp[i - 1] and s[i:j+1] in wordDict:
                    dp[j] = True
                    break

        return dp[len(s) - 1]

```

小結

常見處理方式是給 0 位置佔位，這樣處理問題時一視同仁，初始化則在原來基礎上 length+1，返回結果 f[n]

- 狀態可以為前 i 個
- 初始化 length+1
- 取值 index=i-1
- 返回值：f[n]或者 f[m][n]

## Two Sequences DP（40%）

### [longest-common-subsequence](https://leetcode.com/problems/longest-common-subsequence/)

> 給定兩個字符串  text1 和  text2，返回這兩個字符串的最長公共子序列。
> 一個字符串的   子序列   是指這樣一個新的字符串：它是由原字符串在不改變字符的相對順序的情況下刪除某些字符（也可以不刪除任何字符）後組成的新字符串。
> 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。兩個字符串的「公共子序列」是這兩個字符串所共同擁有的子序列。

- 二維 DP 若隻與當前行和上一行有關，可將空間複雜度降到線性

```Python
class Solution:
    def longestCommonSubsequence(self, t1: str, t2: str) -> int:

        if t1 == '' or t2 == '':
            return 0

        if len(t1) < len(t2):
            t1, t2 = t2, t1

        dp = [int(t2[0] == t1[0])] * len(t2) # previous row
        dp_new = [0] * len(t2) # current row

        for j in range(1, len(t2)):
            dp[j] = 1 if t2[j] == t1[0] else dp[j - 1]

        for i in range(1, len(t1)):
            dp_new[0] = 1 if dp[0] == 1 or t2[0] == t1[i] else 0
            for j in range(1, len(t2)):
                if t2[j] != t1[i]:
                    dp_new[j] = max(dp[j], dp_new[j - 1])
                else:
                    dp_new[j] = dp[j - 1] + 1
            dp, dp_new = dp_new, dp

        return dp[-1]
```

### [edit-distance](https://leetcode.com/problems/edit-distance/)

> 給你兩個單詞  word1 和  word2，請你計算出將  word1  轉換成  word2 所使用的最少操作數  
> 你可以對一個單詞進行如下三種操作：
> 插入一個字符
> 刪除一個字符
> 替換一個字符

思路：和上題很類似，相等則不需要操作，否則取刪除、插入、替換最小操作次數的值+1

```Python
class Solution:
    def minDistance(self, w1: str, w2: str) -> int:

        if w1 == '': return len(w2)
        if w2 == '': return len(w1)

        m, n = len(w1), len(w2)
        if m < n:
            w1, w2, m, n = w2, w1, n, m

        dp = [int(w1[0] != w2[0])] * n
        dp_new = [0] * n

        for j in range(1, n):
            dp[j] = dp[j - 1] + int(w2[j] != w1[0] or dp[j - 1] != j)

        for i in range(1, m):
            dp_new[0] = dp[0] + int(w2[0] != w1[i] or dp[0] != i)

            for j in range(1, n):
                dp_new[j] = min(dp[j - 1] + int(w2[j] != w1[i]), dp[j] + 1, dp_new[j - 1] + 1)

            dp, dp_new = dp_new, dp


        return dp[-1]
```

說明

> 另外一種做法：MAXLEN(a,b)-LCS(a,b)

## 零錢和背包（10%）

### [coin-change](https://leetcode.com/problems/coin-change/)

> 給定不同麵額的硬幣 coins 和一個總金額 amount。編寫一個函數來計算可以湊成總金額所需的最少的硬幣個數。如果有有任何一種硬幣組合能組成總金額，返回  -1。

思路：和其他 DP 不太一樣，i 錶示錢或者容量

```Python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:

        dp = [0] * (amount + 1)

        for i in range(1, len(dp)):
            dp[i] = float('inf')

            for coin in coins:
                if i >= coin and dp[i - coin] + 1 < dp[i]:
                    dp[i] = dp[i - coin] + 1

        return -1 if dp[amount] == float('inf') else dp[amount]
```

### [backpack](https://www.lintcode.com/problem/backpack/description)

> 在 n 個物品中挑選若幹物品裝入背包，最多能裝多滿？假設背包的大小為 m，每個物品的大小為 A[i]

```Python
class Solution:
    def backPack(self, m, A):

        n = len(A)

        dp = [0] * (m + 1)
        dp_new = [0] * (m + 1)

        for i in range(n):
            for j in range(1, m + 1):
                use_Ai = 0 if j - A[i] < 0 else dp[j - A[i]] + A[i]
                dp_new[j] = max(dp[j], use_Ai)

            dp, dp_new = dp_new, dp

        return dp[-1]

```

### [backpack-ii](https://www.lintcode.com/problem/backpack-ii/description)

> 有 `n` 個物品和一個大小為 `m` 的背包. 給定數組 `A` 錶示每個物品的大小和數組 `V` 錶示每個物品的價值.
> 問最多能裝入背包的總價值是多大?

思路：dp(i, j) 為前 i 個物品，裝入 j 背包的最大價值

```Python
class Solution:
    def backPackII(self, m, A, V):

        n = len(A)

        dp = [0] * (m + 1)
        dp_new = [0] * (m + 1)

        for i in range(n):
            for j in range(1, m + 1):
                use_Ai = 0 if j - A[i] < 0 else dp[j - A[i]] + V[i] # previous problem is a special case of this problem that V(i) = A(i)
                dp_new[j] = max(dp[j], use_Ai)

            dp, dp_new = dp_new, dp

        return dp[-1]

```

## 補充

### [maximum-product-subarray](https://leetcode.com/problems/maximum-product-subarray/)

> 最大乘積子串

處理負數情況稍微有點複雜，註意需要同時 DP 正數乘積和負數乘積

```Python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:

        max_product = float('-inf')

        dp_pos, dp_neg = 0, 0

        for num in nums:
            if num > 0:
                dp_pos, dp_neg = max(num, num * dp_pos), dp_neg * num
            else:
                dp_pos, dp_neg = dp_neg * num, min(num, dp_pos * num)

            if dp_pos != 0:
                max_product = max(max_product, dp_pos)
            elif dp_neg != 0:
                max_product = max(max_product, dp_neg)
            else:
                max_product = max(max_product, 0)

        return max_product
```

### [decode-ways](https://leetcode.com/problems/decode-ways/)

> 1 到 26 分別對應 a 到 z，給定輸入數字串，問總共有多少種譯碼方法

常規 DP 題，註意處理 edge case 即可

```Python
class Solution:
    def numDecodings(self, s: str) -> int:

        def valid_2(i):
            if i < 1:
                return 0
            num = int(s[i-1:i+1])
            return int(num > 9 and num < 27)

        dp_1, dp_2 = 1, 0
        for i in range(len(s)):
            dp_1, dp_2 = dp_1 * int(s[i] != '0') + dp_2 * valid_2(i), dp_1

        return dp_1
```

### [best-time-to-buy-and-sell-stock-with-cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

> 給定股票每天的價格，每天可以買入賣出，買入後必須賣出才可以進行下一次購買，賣出後一天不可以購買，問可以獲得的最大利潤

經典的維特比譯碼類問題，找到狀態空間和狀態轉移關係即可

```Python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:

        buy, buy_then_nothing, sell, sell_then_nothing = float('-inf'), float('-inf'), float('-inf'), 0

        for p in prices:
            buy, buy_then_nothing, sell, sell_then_nothing = sell_then_nothing - p, max(buy, buy_then_nothing), max(buy, buy_then_nothing) + p, max(sell, sell_then_nothing)

        return max(buy, buy_then_nothing, sell, sell_then_nothing)
```

### [word-break-ii](https://leetcode.com/problems/word-break-ii/)

> 給定字符串和可選的單詞列錶，求字符串所有的分割方式

思路：此題 DP 解法容易想但並不是好做法，因為和 word-break 不同，此題需要返回所有可行分割而不是找到一組就可以。這裏使用 個人推薦 backtrack with memoization。

```Python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:

        n = len(s)
        result = []
        mem = collections.defaultdict(list)
        wordDict = set(wordDict)

        def backtrack(first=0, route=[]):
            if first == n:
                result.append(' '.join(route))
                return True

            if first not in mem:
                for next_first in range(first + 1, n + 1):
                    if s[first:next_first] in wordDict:
                        route.append(s[first:next_first])
                        if backtrack(next_first, route):
                            mem[first].append(next_first)
                        route.pop()
                if len(mem[first]) > 0:
                    return True
            elif len(mem[first]) > 0:
                for next_first in mem[first]:
                    route.append(s[first:next_first])
                    backtrack(next_first)
                    route.pop()
                return True

            return False

        backtrack()
        return result
```

### [burst-balloons](https://leetcode.com/problems/burst-balloons/)

> n 個氣球排成一行，每個氣球上有一個分數，每次戳爆一個氣球得分為該氣球分數和相鄰兩氣球分數的乘積，求最大得分

此題主要難點是構造 DP 的狀態，過程為逆著氣球戳爆的順序

```Python
class Solution:
    def maxCoins(self, nums: List[int]) -> int:

        n = len(nums)
        nums.append(1)
        dp = [[0] * (n + 1) for _ in range(n + 1)]

        for dist in range(2, n + 2):
            for left in range(-1, n - dist + 1):
                right = left + dist
                max_coin = float('-inf')
                left_right = nums[left] * nums[right]
                for j in range(left + 1, right):
                    max_coin = max(max_coin, left_right * nums[j] + dp[left][j] + dp[j][right])
                dp[left][right] = max_coin
        nums.pop()
        return dp[-1][n]
```

## 練習

Matrix DP (10%)

- [ ] [triangle](https://leetcode.com/problems/triangle/)
- [ ] [minimum-path-sum](https://leetcode.com/problems/minimum-path-sum/)
- [ ] [unique-paths](https://leetcode.com/problems/unique-paths/)
- [ ] [unique-paths-ii](https://leetcode.com/problems/unique-paths-ii/)

Sequence (40%)

- [ ] [climbing-stairs](https://leetcode.com/problems/climbing-stairs/)
- [ ] [jump-game](https://leetcode.com/problems/jump-game/)
- [ ] [jump-game-ii](https://leetcode.com/problems/jump-game-ii/)
- [ ] [palindrome-partitioning-ii](https://leetcode.com/problems/palindrome-partitioning-ii/)
- [ ] [longest-increasing-subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)
- [ ] [word-break](https://leetcode.com/problems/word-break/)

Two Sequences DP (40%)

- [ ] [longest-common-subsequence](https://leetcode.com/problems/longest-common-subsequence/)
- [ ] [edit-distance](https://leetcode.com/problems/edit-distance/)

Backpack & Coin Change (10%)

- [ ] [coin-change](https://leetcode.com/problems/coin-change/)
- [ ] [backpack](https://www.lintcode.com/problem/backpack/description)
- [ ] [backpack-ii](https://www.lintcode.com/problem/backpack-ii/description)
