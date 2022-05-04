# 鏈結串列

## 基本技能

鏈結串列相關的核心點

- null/nil 異常處理
- dummy node 啞巴節點
- 快慢指針
- 插入一個節點到排序鏈結串列
- 從一個鏈結串列中移除一個節點
- 翻轉鏈結串列
- 合並兩個鏈結串列
- 找到鏈結串列的中間節點

## 常見題型

### [remove-duplicates-from-sorted-list](https://leetcode.com/problems/remove-duplicates-from-sorted-list/)

> 給定一個排序鏈結串列，刪除所有重複的元素，使得每個元素隻出現一次。

```Python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:

        if head is None:
            return head

        current = head

        while current.next is not None:
            if current.next.val == current.val:
                current.next = current.next.next
            else:
                current = current.next

        return head
```

### [remove-duplicates-from-sorted-list-ii](https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/)

> 給定一個排序鏈結串列，刪除所有含有重複數字的節點，隻保留原始鏈結串列中   冇有重複出現的數字。

- 思路：鏈結串列頭結點可能被刪除，所以用 dummy node 輔助刪除

```Python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:

        if head is None:
            return head

        dummy = ListNode(next=head)

        current, peek = dummy, head
        find_dup = False
        while peek.next is not None:
            if peek.next.val == peek.val:
                find_dup = True
                peek.next = peek.next.next
            else:
                if find_dup:
                    find_dup = False
                    current.next = current.next.next
                else:
                    current = current.next
                peek = peek.next

        if find_dup:
            current.next = current.next.next

        return dummy.next
```

註意點
• A->B->C 刪除 B，A.next = C
• 刪除用一個 Dummy Node 節點輔助（允許頭節點可變）
• 訪問 X.next 、X.value 一定要保證 X != nil

### [reverse-linked-list](https://leetcode.com/problems/reverse-linked-list/)

> 反轉一個單鏈結串列。

- 思路：將當前結點放置到頭結點

```Python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:

        if head is None:
            return head

        tail = head
        while tail.next is not None:
            # put tail.next to head
            tmp = tail.next
            tail.next = tail.next.next
            tmp.next = head
            head = tmp

        return head
```

- Recursive method is tricky

```Python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:

        if head is None or head.next is None:
            return head

        rev_next = self.reverseList(head.next)
        head.next.next = head
        head.next = None

        return rev_next
```

### [reverse-linked-list-ii](https://leetcode.com/problems/reverse-linked-list-ii/)

> 反轉從位置  *m*  到  *n*  的鏈結串列。請使用一趟掃描完成反轉。

- 思路：先找到 m 處, 再反轉 n - m 次即可

```Python
class Solution:
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:

        if head is None:
            return head

        n -= m # number of times of reverse

        curr = dummy = ListNode(next=head)
        while m > 1: # find node at m - 1
            curr = curr.next
            m -= 1

        start = curr.next
        while n > 0: # reverse n - m times
            tmp = start.next
            start.next = tmp.next
            tmp.next = curr.next
            curr.next = tmp
            n -= 1
        return dummy.next
```

### [merge-two-sorted-lists](https://leetcode.com/problems/merge-two-sorted-lists/)

> 將兩個升序鏈結串列合並為一個新的升序鏈結串列並返回。新鏈結串列是通過拚接給定的兩個鏈結串列的所有節點組成的。

- 思路：通過 dummy node 鏈結串列，連接各個元素

```Python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:

        tail = dummy = ListNode()
        while l1 is not None and l2 is not None:
            if l1.val > l2.val:
                tail.next = l2
                l2 = l2.next
            else:
                tail.next = l1
                l1 = l1.next
            tail = tail.next

        if l1 is None:
            tail.next = l2
        else:
            tail.next = l1

        return dummy.next
```

### [partition-list](https://leetcode.com/problems/partition-list/)

> 給定一個鏈結串列和一個特定值 x，對鏈結串列進行分隔，使得所有小於  *x*  的節點都在大於或等於  *x*  的節點之前。

- 思路：將大於 x 的節點，放到另外一個鏈結串列，最後連接這兩個鏈結串列

```go
class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:

        p = l = ListNode()
        q = s = ListNode(next=head)

        while q.next is not None:
            if q.next.val < x:
                q = q.next
            else:
                p.next = q.next
                q.next = q.next.next
                p = p.next

        p.next = None
        q.next = l.next

	return s.next
```

啞巴節點使用場景

> 當頭節點不確定的時候，使用啞巴節點

### [sort-list](https://leetcode.com/problems/sort-list/)

> 在  *O*(*n* log *n*) 時間複雜度和常數級空間複雜度下，對鏈結串列進行排序。

- 思路：歸並排序，slow-fast 找中點

```Python
class Solution:

    def _merge(self, l1, l2):
        tail = l_merge = ListNode()

        while l1 is not None and l2 is not None:
            if l1.val > l2.val:
                tail.next = l2
                l2 = l2.next
            else:
                tail.next = l1
                l1 = l1.next
            tail = tail.next

        if l1 is not None:
            tail.next = l1
        else:
            tail.next = l2

        return l_merge.next

    def _findmid(self, head):
        slow, fast = head, head.next
        while fast is not None and fast.next is not None:
            fast = fast.next.next
            slow = slow.next

        return slow

    def sortList(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head

        mid = self._findmid(head)
        tail = mid.next
        mid.next = None # break from middle

        return self._merge(self.sortList(head), self.sortList(tail))
```

註意點

- 快慢指針 判斷 fast 及 fast.Next 是否為 nil 值
- 遞迴 mergeSort 需要斷開中間節點
- 遞迴返回條件為 head 為 nil 或者 head.Next 為 nil

### [reorder-list](https://leetcode.com/problems/reorder-list/)

> 給定一個單鏈結串列  *L*：*L*→*L*→…→*L\_\_n*→*L*
> 將其重新排列後變為： *L*→*L\_\_n*→*L*→*L\_\_n*→*L*→*L\_\_n*→…

- 思路：找到中點斷開，翻轉後麵部分，然後合並前後兩個鏈結串列

```Python
class Solution:

    def reverseList(self, head: ListNode) -> ListNode:

        prev, curr = None, head

        while curr is not None:
            curr.next, prev, curr = prev, curr, curr.next

        return prev

    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if head is None or head.next is None or head.next.next is None:
            return

        slow, fast = head, head.next
        while fast is not None and fast.next is not None:
            fast = fast.next.next
            slow = slow.next

        h, m = head, slow.next
        slow.next = None

        m = self.reverseList(m)

        while h is not None and m is not None:
            p = m.next
            m.next = h.next
            h.next = m
            h = h.next.next
            m = p

        return
```

### [linked-list-cycle](https://leetcode.com/problems/linked-list-cycle/)

> 給定一個鏈結串列，判斷鏈結串列中是否有環。

- 思路 1：Hash Table 記錄所有結點判斷重複，空間複雜度 O(n) 非最優，時間複雜度 O(n) 但必然需要 n 次循環
- 思路 2：快慢指針，快慢指針相同則有環，證明：如果有環每走一步快慢指針距離會減 1，空間複雜度 O(1) 最優，時間複雜度 O(n) 但循環次數小於等於 n
  ![fast_slow_linked_list](https://img.fuiboom.com/img/fast_slow_linked_list.png)

```Python
class Solution:
    def hasCycle(self, head: ListNode) -> bool:

        slow = fast = head

        while fast is not None and fast.next is not None:
            slow = slow.next
	    fast = fast.next.next
            if fast == slow:
                return True

        return False
```

### [linked-list-cycle-ii](https://leetcode.com/problems/linked-list-cycle-ii/)

> 給定一個鏈結串列，返回鏈結串列開始入環的第一個節點。  如果鏈結串列無環，則返回  `null`。

- 思路：快慢指針，快慢相遇之後，慢指針回到頭，快慢指針步調一緻一起移動，相遇點即為入環點。

![cycled_linked_list](https://img.fuiboom.com/img/cycled_linked_list.png)

```Python
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:

        slow = fast = head

        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next

            if slow == fast:
                slow = head
                while fast != slow:
                    fast = fast.next
                    slow = slow.next
                return slow

        return None
```

坑點

- 指針比較時直接比較對象，不要用值比較，鏈結串列中有可能存在重複值情況
- 第一次相交後，快指針需要從下一個節點開始和頭指針一起勻速移動

註意，此題中使用 slow = fast = head 是為了保證最後找環起始點時移動步數相同，但是作為找中點使用時**一般用 fast=head.Next 較多**，因為這樣可以知道中點的上一個節點，可以用來刪除等操作。

- fast 如果初始化為 head.Next 則中點在 slow.Next
- fast 初始化為 head,則中點在 slow

### [palindrome-linked-list](https://leetcode.com/problems/palindrome-linked-list/)

> 請判斷一個鏈結串列是否為回文鏈結串列。

- 思路：O(1) 空間複雜度的解法需要破壞原鏈結串列（找中點 -> 反轉後半個 list -> 判斷回文），在實際應用中往往還需要複原（後半個 list 再反轉一次後拚接），操作比較複雜，這裏給出更專案化的做法

```Python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:

        s = []
        slow = fast = head
        while fast is not None and fast.next is not None:
            s.append(slow.val)
            slow = slow.next
            fast = fast.next.next

        if fast is not None:
            slow = slow.next

        while len(s) > 0:
            if slow.val != s.pop():
                return False
            slow = slow.next

        return True
```

### [copy-list-with-random-pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)

> 給定一個鏈結串列，每個節點包含一個額外增加的隨機指針，該指針可以指嚮鏈結串列中的任何節點或空節點。
> 要求返回這個鏈結串列的 深拷貝。

- 思路 1：hash table 存儲 random 指針的連接關係

```Python
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':

        if head is None:
            return None

        parent = collections.defaultdict(list)

        out = Node(0)
        o, n = head, out
        while o is not None:
            n.next = Node(o.val)
            n = n.next
            if o.random is not None:
                parent[o.random].append(n)
            o = o.next

        o, n = head, out.next
        while o is not None:
            if o in parent:
                for p in parent[o]:
                    p.random = n
            o = o.next
            n = n.next

        return out.next
```

- 思路 2：複製結點跟在原結點後麵，間接維護連接關係，優化空間複雜度，建立好新 list 的 random 鏈接後分離

```Python
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':

        if head is None:
            return None

        p = head
        while p is not None:
            p.next = Node(p.val, p.next)
            p = p.next.next

        p = head
        while p is not None:
            if p.random is not None:
                p.next.random = p.random.next
            p = p.next.next

        new = head.next
        o, n = head, new
        while n.next is not None:
            o.next = n.next
            n.next = n.next.next
            o = o.next
            n = n.next
        o.next = None

        return new
```

## 總結

鏈結串列必須要掌握的一些點，通過下麵練習題，基本大部分的鏈結串列類的題目都是手到擒來~

- null/nil 異常處理
- dummy node 啞巴節點
- 快慢指針
- 插入一個節點到排序鏈結串列
- 從一個鏈結串列中移除一個節點
- 翻轉鏈結串列
- 合並兩個鏈結串列
- 找到鏈結串列的中間節點

## 練習

- [ ] [remove-duplicates-from-sorted-list](https://leetcode.com/problems/remove-duplicates-from-sorted-list/)
- [ ] [remove-duplicates-from-sorted-list-ii](https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/)
- [ ] [reverse-linked-list](https://leetcode.com/problems/reverse-linked-list/)
- [ ] [reverse-linked-list-ii](https://leetcode.com/problems/reverse-linked-list-ii/)
- [ ] [merge-two-sorted-lists](https://leetcode.com/problems/merge-two-sorted-lists/)
- [ ] [partition-list](https://leetcode.com/problems/partition-list/)
- [ ] [sort-list](https://leetcode.com/problems/sort-list/)
- [ ] [reorder-list](https://leetcode.com/problems/reorder-list/)
- [ ] [linked-list-cycle](https://leetcode.com/problems/linked-list-cycle/)
- [ ] [linked-list-cycle-ii](https://leetcode.com/problems/https://leetcode.com/problems/linked-list-cycle-ii/)
- [ ] [palindrome-linked-list](https://leetcode.com/problems/palindrome-linked-list/)
- [ ] [copy-list-with-random-pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)
