# 遞迴

## 介紹

將大問題轉化為小問題，通過遞迴依次解決各個小問題

## 示例

### [reverse-string](https://leetcode.com/problems/reverse-string/)

> 編寫一個函數，其作用是將輸入的字符串反轉過來。輸入字符串以字符數組  `char[]`  的形式給出。

```Python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        def rev_rec(s, i, j):
            if i >= j:
                return
            s[i], s[j] = s[j], s[i]
            rev_rec(s, i + 1, j - 1)
            return

        rev_rec(s, 0, len(s) - 1)

        return
```

### [swap-nodes-in-pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)

> 給定一個鏈結串列，兩兩交換其中相鄰的節點，並返回交換後的鏈結串列。
> **你不能隻是單純的改變節點內部的值**，而是需要實際的進行節點交換。

```Python
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:

        if head is not None and head.next is not None:
            head_next_pair = self.swapPairs(head.next.next)
            p = head.next
            head.next = head_next_pair
            p.next = head
            head = p

        return head
```

### [unique-binary-search-trees-ii](https://leetcode.com/problems/unique-binary-search-trees-ii/)

> 給定一個整數 n，生成所有由 1 ... n 為節點所組成的二元搜尋樹。

註意：此題用來訓練遞迴思維有理論意義，但是實際上算法返回的樹並不是 deep copy，多個樹之間會共享子樹。

```Python
class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:

        def generateTrees_rec(i, j):

            if i > j:
                return [None]

            result = []
            for m in range(i, j + 1):
                left = generateTrees_rec(i, m - 1)
                right = generateTrees_rec(m + 1, j)

                for l in left:
                    for r in right:
                        result.append(TreeNode(m, l, r))

            return result

        return generateTrees_rec(1, n) if n > 0 else []
```

## 遞迴 + 備忘錄 (recursion with memorization, top-down DP)

### [fibonacci-number](https://leetcode.com/problems/fibonacci-number/)

> 斐波那契數，通常用  F(n) 錶示，形成的序列稱為斐波那契數列。該數列由  0 和 1 開始，後麵的每一項數字都是前麵兩項數字的和。也就是：
> F(0) = 0,   F(1) = 1
> F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
> 給定  N，計算  F(N)。

```Python
class Solution:
    def fib(self, N: int) -> int:

        mem = [-1] * (N + 2)

        mem[0], mem[1] = 0, 1

        def fib_rec(n):
            if mem[n] == -1:
                mem[n] = fib_rec(n - 1) + fib_rec(n - 2)
            return mem[n]

        return fib_rec(N)
```

## 練習

- [ ] [reverse-string](https://leetcode.com/problems/reverse-string/)
- [ ] [swap-nodes-in-pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)
- [ ] [unique-binary-search-trees-ii](https://leetcode.com/problems/unique-binary-search-trees-ii/)
- [ ] [fibonacci-number](https://leetcode.com/problems/fibonacci-number/)
