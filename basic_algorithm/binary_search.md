# 二分搜尋

給一個**有序數組**和目標值，找第一次/最後一次/任何一次出現的索引，時間複雜度 O(logN)。

## 樣板

常用的二分搜尋樣板有如下三種形式：

![binary_search_template](https://img.fuiboom.com/img/binary_search_template.png)

其中，樣板 1 和 3 是最常用的，幾乎所有二分查找問題都可以用其中之一輕鬆實現。樣板 2 更高級一些，用於解決某些類型的問題。詳細的對比可以參考 Leetcode 上的文章：[二分搜尋樣板](https://leetcode.com/explore/learn/card/binary-search/212/template-analysis/847/)。

### [binary-search](https://leetcode.com/problems/binary-search/)

> 給定一個  n  個元素有序的（升序）整型數組  nums 和一個目標值  target  ，寫一個函數搜索  nums  中的 target，如果目標值存在返回下標，否則返回 -1。

- 樣板 3 的實現

```Python
class Solution:
    def search(self, nums: List[int], target: int) -> int:

        l, r = 0, len(nums) - 1

        while l + 1 < r:
            mid = l + (r - l) // 2
            if nums[mid] < target:
                l = mid
            else:
                r = mid

        if nums[l] == target:
            return l
        elif nums[r] == target:
            return r
        else:
            return -1
```

- 如果是最簡單的二分搜尋，不需要找第一個、最後一個位置，或者是有有重複元素，可以使用樣板 1，代碼更簡潔。同時，如果搜索失敗，left 是第一個大於 target 的索引，right 是最後一個小於 target 的索引。

```Python
class Solution:
    def search(self, nums: List[int], target: int) -> int:

        l, r = 0, len(nums) - 1

        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                r = mid - 1
            else:
                l = mid + 1

        return -1
```

- 樣板 2 的實現

```Python
class Solution:
    def search(self, nums: List[int], target: int) -> int:

        l, r = 0, len(nums) - 1

        while l < r:
            mid = l + (r - l) // 2
            if nums[mid] < target:
                l = mid + 1
            else:
                r = mid

        if nums[l] == target:
            return l

        return -1
```

## 常見題目

### [find-first-and-last-position-of-element-in-sorted-array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

> 給定一個包含 n 個整數的排序數組，找出給定目標值 target 的起始和結束位置。如果目標值不在數組中，則返回`[-1, -1]`

- 思路：核心點就是找第一個 target 的索引，和最後一個 target 的索引，所以用兩次二分搜尋分別找第一次和最後一次的位置，下麵是使用樣板 3 的解法

```Python
class Solution:
    def searchRange(self, nums, target):
        Range = [-1, -1]
        if len(nums) == 0:
            return Range

        l, r = 0, len(nums) - 1
        while l + 1 < r:
            mid = l + (r - l) // 2
            if nums[mid] < target:
                l = mid
            else:
                r = mid

        if nums[l] == target:
            Range[0] = l
        elif nums[r] == target:
            Range[0] = r
        else:
            return Range

        l, r = 0, len(nums) - 1
        while l + 1 < r:
            mid = l + (r - l) // 2
            if nums[mid] <= target:
                l = mid
            else:
                r = mid

        if nums[r] == target:
            Range[1] = r
        else:
            Range[1] = l

        return Range
```

- 使用樣板 2 的解法

```Python
class Solution:
    def searchRange(self, nums, target):
        Range = [-1, -1]
        if len(nums) == 0:
            return Range

        l, r = 0, len(nums) - 1
        while l < r:
            mid = l + (r - l) // 2
            if nums[mid] < target:
                l = mid + 1
            else:
                r = mid

        if nums[l] == target:
            Range[0] = l
        else:
            return Range

        l, r = 0, len(nums) - 1
        while l < r:
            mid = l + (r - l + 1) // 2
            if nums[mid] > target:
                r = mid - 1
            else:
                l = mid

        Range[1] = l
        return Range
```

### [search-insert-position](https://leetcode.com/problems/search-insert-position/)

> 給定一個排序數組和一個目標值，在數組中找到目標值，並返回其索引。如果目標值不存在於數組中，返回它將會被按順序插入的位置。

- 使用樣板 1，若不存在，左邊界為第一個大於目標值的索引（插入位置），右邊界為最後一個小於目標值的索引

```Python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:

        l, r = 0, len(nums) - 1

        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                r = mid - 1
            else:
                l = mid + 1

        return l
```

### [search-a-2d-matrix](https://leetcode.com/problems/search-a-2d-matrix/)

> 編寫一個高效的算法來判斷  m x n  矩陣中，是否存在一個目標值。該矩陣具有如下特性：
>
> 1. 每行中的整數從左到右按升序排列。
>
> 2. 每行的第一個整數大於前一行的最後一個整數。

- 兩次二分，首先定位行數，接著定位列數

```Python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:

        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False

        l, r = 0, len(matrix) - 1

        while l <= r:
            mid = l + (r - l) // 2
            if matrix[mid][0] == target:
                return True
            elif matrix[mid][0] < target:
                l = mid + 1
            else:
                r = mid - 1

        row = r
        l, r = 0, len(matrix[0]) - 1
        while l <= r:
            mid = l + (r - l) // 2
            if matrix[row][mid] == target:
                return True
            elif matrix[row][mid] < target:
                l = mid + 1
            else:
                r = mid - 1

        return False
```

### [find-minimum-in-rotated-sorted-array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)

> 假設按照升序排序的數組在預先未知的某個點上進行了旋轉，例如，數組 [0, 1, 2, 4, 5, 6, 7] 可能變為 [4, 5, 6, 7, 0, 1, 2]。請找出其中最小的元素。假設數組中無重複元素。

- 使用二分搜尋，當中間元素大於右側元素時意味著拐點即最小元素在右側，否則在左側

```Python
class Solution:
    def findMin(self, nums: List[int]) -> int:

        l , r = 0, len(nums) - 1

        while l < r:
            mid = l + (r - l) // 2
            if nums[mid] > nums[r]: # 數組有重複時，若 nums[l] == nums[mid] == nums[r]，無法判斷移動方向
                l = mid + 1
            else:
                r = mid

        return nums[l]
```

### [find-minimum-in-rotated-sorted-array-ii](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/)

> 假設按照升序排序的數組在預先未知的某個點上進行了旋轉，例如，數組 [0, 1, 2, 4, 5, 6, 7] 可能變為 [4, 5, 6, 7, 0, 1, 2]。請找出其中最小的元素。數組中可能包含重複元素。

```Python
class Solution:
    def findMin(self, nums: List[int]) -> int:

        l , r = 0, len(nums) - 1

        while l < r:
            mid = l + (r - l) // 2
            if nums[mid] > nums[r]:
                l = mid + 1
            elif nums[mid] < nums[r] or nums[mid] != nums[l]:
                r = mid
            else: # nums[l] == nums[mid] == nums[r]
                l += 1

        return nums[l]
```

### [search-in-rotated-sorted-array](https://leetcode.com/problems/search-in-rotated-sorted-array/)

> 假設按照升序排序的數組在預先未知的某個點上進行了旋轉，例如，數組 [0, 1, 2, 4, 5, 6, 7] 可能變為 [4, 5, 6, 7, 0, 1, 2]。搜索一個給定的目標值，如果數組中存在這個目標值，則返回它的索引，否則返回  -1。假設數組中不存在重複的元素。

```Python
class Solution:
    def search(self, nums: List[int], target: int) -> int:

        l , r = 0, len(nums) - 1

        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                if nums[l] > target and nums[mid] > nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
            else:
                if nums[r] < target and nums[mid] < nums[l]:
                    r = mid - 1
                else:
                    l = mid + 1
        return -1
```

### [search-in-rotated-sorted-array-ii](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/)

> 假設按照升序排序的數組在預先未知的某個點上進行了旋轉，例如，數組 [0, 0, 1, 2, 2, 5, 6] 可能變為 [2, 5, 6, 0, 0, 1, 2]。編寫一個函數來判斷給定的目標值是否存在於數組中，若存在返回  true，否則返回  false。數組中可能包含重複元素。

```Python
class Solution:
    def search(self, nums: List[int], target: int) -> int:

        l , r = 0, len(nums) - 1

        while l <= r:
            if nums[l] == nums[r] and nums[l] != target:
                l += 1
                r -= 1
                continue
            mid = l + (r - l) // 2
            if nums[mid] == target:
                return True
            elif nums[mid] > target:
                if nums[l] > target and nums[mid] > nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
            else:
                if nums[r] < target and nums[mid] < nums[l]:
                    r = mid - 1
                else:
                    l = mid + 1
        return False
```

## 隱含的二分搜尋

有時用到二分搜尋的題目並不會直接給你一個有序數組，它隱含在題目中，需要你去發現或者構造。一類常見的隱含的二分搜尋的問題是求某個有界數據的最值，以最小值為例，當數據比最小值大時都符合條件，比最小值小時都不符合條件，那麼符合/不符合條件就構成了一種有序關係，再加上數據有界，我們就可以使用二分搜尋來找數據的最小值。註意，數據的界一般也不會在題目中明確提示你，需要你自己去發現。

### [koko-eating-bananas](https://leetcode.com/problems/koko-eating-bananas/)

```Python
class Solution:
    def minEatingSpeed(self, piles: List[int], H: int) -> int:

        l, r = 1, max(piles)

        while l < r:
            mid = l + (r - l) // 2
            if sum([-pile // mid for pile in piles]) < -H:
                l = mid + 1
            else:
                r = mid

        return l
```

## 總結

二分搜尋核心四點要素（必背&理解）

- 1、初始化：start=0、end=len-1
- 2、循環退出條件：start + 1 < end
- 3、比較中點和目標值：A[mid] ==、 <、> target
- 4、判斷最後兩個元素是否符合：A[start]、A[end] ? target

## 練習題

- [ ] [search-for-range](https://www.lintcode.com/problem/search-for-a-range/description)
- [ ] [search-insert-position](https://leetcode.com/problems/search-insert-position/)
- [ ] [search-a-2d-matrix](https://leetcode.com/problems/search-a-2d-matrix/)
- [ ] [first-bad-version](https://leetcode.com/problems/first-bad-version/)
- [ ] [find-minimum-in-rotated-sorted-array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)
- [ ] [find-minimum-in-rotated-sorted-array-ii](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/)
- [ ] [search-in-rotated-sorted-array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
- [ ] [search-in-rotated-sorted-array-ii](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/)
