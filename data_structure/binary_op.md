# 二進位

## 常見二進位操作

### 基本操作

a=0^a=a^0

0=a^a

由上麵兩個推導出：a=a^b^b

### 交換兩個數

a=a^b

b=a^b

a=a^b

### 移除最後一個 1

a=n&(n-1)

### 獲取最後一個 1

diff=(n&(n-1))^n

## 常見題目

### [single-number](https://leetcode.com/problems/single-number/)

> 給定一個**非空**整數數組，除了某個元素隻出現一次以外，其餘每個元素均出現兩次。找出那個隻出現了一次的元素。

```Python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:

        out = 0
        for num in nums:
            out ^= num

        return out
```

### [single-number-ii](https://leetcode.com/problems/single-number-ii/)

> 給定一個**非空**整數數組，除了某個元素隻出現一次以外，其餘每個元素均出現了三次。找出那個隻出現了一次的元素。

```Python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        seen_once = seen_twice = 0

        for num in nums:
            seen_once = ~seen_twice & (seen_once ^ num)
            seen_twice = ~seen_once & (seen_twice ^ num)

        return seen_once
```

### [single-number-iii](https://leetcode.com/problems/single-number-iii/)

> 給定一個整數數組  `nums`，其中恰好有兩個元素隻出現一次，其餘所有元素均出現兩次。 找出隻出現一次的那兩個元素。

```Python
class Solution:
    def singleNumber(self, nums: int) -> List[int]:
        # difference between two numbers (x and y) which were seen only once
        bitmask = 0
        for num in nums:
            bitmask ^= num

        # rightmost 1-bit diff between x and y
        diff = bitmask & (-bitmask)

        x = 0
        for num in nums:
            # bitmask which will contain only x
            if num & diff:
                x ^= num

        return [x, bitmask^x]
```

### [number-of-1-bits](https://leetcode.com/problems/number-of-1-bits/)

> 編寫一個函數，輸入是一個無符號整數，返回其二進位錶達式中數字位數為 ‘1’  的個數（也被稱為[漢明重量](https://baike.baidu.com/item/%E6%B1%89%E6%98%8E%E9%87%8D%E9%87%8F)）。

```Python
class Solution:
    def hammingWeight(self, n: int) -> int:
        num_ones = 0
        while n > 0:
            num_ones += 1
            n &= n - 1
        return num_ones
```

### [counting-bits](https://leetcode.com/problems/counting-bits/)

> 給定一個非負整數  **num**。對於  0 ≤ i ≤ num  範圍中的每個數字  i ，計算其二進位數中的 1 的數目並將它們作為數組返回。

- 思路：利用上一題的解法容易想到 O(nk) 的解法，k 為位數。但是實際上可以利用動態規劃將複雜度降到 O(n)，想法其實也很簡單，即當前數的 1 個數等於比它少一個 1 的數的結果加 1。下麵給出三種 DP 解法

```Python
# x <- x // 2
class Solution:
    def countBits(self, num: int) -> List[int]:

        num_ones = [0] * (num + 1)

        for i in range(1, num + 1):
            num_ones[i] = num_ones[i >> 1] + (i & 1) # 註意位運算的優先級

        return num_ones
```

```Python
# x <- x minus right most 1
class Solution:
    def countBits(self, num: int) -> List[int]:

        num_ones = [0] * (num + 1)

        for i in range(1, num + 1):
            num_ones[i] = num_ones[i & (i - 1)] + 1

        return num_ones
```

```Python
# x <- x minus left most 1
class Solution:
    def countBits(self, num: int) -> List[int]:

        num_ones = [0] * (num + 1)

        left_most = 1

        while left_most <= num:
            for i in range(left_most):
                if i + left_most > num:
                    break
                num_ones[i + left_most] = num_ones[i] + 1
            left_most <<= 1

        return num_ones
```

### [reverse-bits](https://leetcode.com/problems/reverse-bits/)

> 顛倒給定的 32 位無符號整數的二進位位。

思路：簡單想法依次顛倒即可。更高級的想法是考慮到處理超長位元串時可能出現重複的 pattern，此時如果使用 cache 記錄出現過的 pattern 並在重複出現時直接調用結果可以節約時間複雜度，具體可以參考 leetcode 給出的解法。

```Python
import functools

class Solution:
    def reverseBits(self, n):
        ret, power = 0, 24
        while n:
            ret += self.reverseByte(n & 0xff) << power
            n = n >> 8
            power -= 8
        return ret

    # memoization with decorator
    @functools.lru_cache(maxsize=256)
    def reverseByte(self, byte):
        return (byte * 0x0202020202 & 0x010884422010) % 1023
```

### [bitwise-and-of-numbers-range](https://leetcode.com/problems/bitwise-and-of-numbers-range/)

> 給定範圍 [m, n]，其中 0 <= m <= n <= 2147483647，返回此範圍內所有數字的按位與（包含 m, n 兩端點）。

思路：直接從 m 到 n 遍歷一遍顯然不是最優。一個性質，如果 m 不等於 n，則結果第一位一定是 0 （中間必定包含一個偶數）。利用這個性質，類似的將 m 和 n 右移後我們也可以判斷第三位、第四位等等，免去了遍歷的時間複雜度。

```Python
class Solution:
    def rangeBitwiseAnd(self, m: int, n: int) -> int:

        shift = 0
        while m < n:
            shift += 1
            m >>= 1
            n >>= 1

        return m << shift
```

## 練習

- [ ] [single-number](https://leetcode.com/problems/single-number/)
- [ ] [single-number-ii](https://leetcode.com/problems/single-number-ii/)
- [ ] [single-number-iii](https://leetcode.com/problems/single-number-iii/)
- [ ] [number-of-1-bits](https://leetcode.com/problems/number-of-1-bits/)
- [ ] [counting-bits](https://leetcode.com/problems/counting-bits/)
- [ ] [reverse-bits](https://leetcode.com/problems/reverse-bits/)
