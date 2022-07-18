# 快速開始

## 數據結構與算法

數據結構是一種數據的表現形式，如鏈結串列、二元樹、堆疊、佇列等都是記憶體中一段數據表現的形式。
算法是一種通用的解決問題的樣板或者思路，大部分數據結構都有一套通用的算法樣板，所以掌握這些通用的算法樣板即可解決各種算法問題。

後面會分專題講解各種數據結構、基本的算法樣板、和一些高級算法樣板，每一個專題都有一些經典練習題，完成所有練習的題後，你對數據結構和算法會有新的收獲和體會。

先介紹兩個算法題，試試感覺~

### [示例 1：strStr](https://leetcode.com/problems/implement-strstr/)

> 給定一個  haystack 字符串和一個 needle 字符串，在 haystack 字符串中找出 needle 字符串出現的第一個位置 (從 0 開始)。如果不存在，則返回  -1。

- 思路：核心點遍歷給定字符串字符，判斷以當前字符開頭字符串是否等於目標字符串

```Python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        L, n = len(needle), len(haystack)

        for start in range(n - L + 1):
            if haystack[start:start + L] == needle:
                return start
        return -1
```

需要注意點

- 循環時，i 不需要到 len-1
- 如果找到目標字符串，len(needle) == j

### [示例 2：subsets](https://leetcode.com/problems/subsets/)

> 給定一組不含重複元素的整數數組 nums，返回該數組所有可能的子集（冪集）。

- 思路：這是一個典型的應用回溯法的題目，簡單來說就是窮盡所有可能性，算法樣板如下

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

- 通過不停的選擇，撤銷選擇，來窮盡所有可能性，最後將滿足條件的結果返回。答案代碼：

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

說明：後面會深入講解幾個典型的回溯算法問題，如果當前不太了解可以暫時先跳過

## 麵試注意點

我們大多數時候，刷算法題可能都是為了準備麵試，所以麵試的時候需要註意一些點

- 快速定位到題目的知識點，找到知識點的**通用樣板**，可能需要根據題目**特殊情況做特殊處理**。
- 先去朝一個解決問題的方向！**先拋出可行解**，而不是最優解！先解決，再優化！
- 代碼的風格要統一，熟悉各類語言的代碼規範。
  - 命名盡量簡潔明了，盡量不用數字命名如：i1、node1、a1、b2
- 常見錯誤總結
  - 訪問下標時，不能訪問越界
  - 空值 nil 問題 run time error

## 練習

- [ ] [strStr](https://leetcode.com/problems/implement-strstr/)
- [ ] [subsets](https://leetcode.com/problems/subsets/)
