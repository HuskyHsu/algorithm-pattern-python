# 二元搜尋樹

## 定義

- 每個節點中的值必須大於（或等於）存儲在其左側子樹中的任何值。
- 每個節點中的值必須小於（或等於）存儲在其右子樹中的任何值。

## 應用

### [validate-binary-search-tree](https://leetcode.com/problems/validate-binary-search-tree/)

> 驗證二元搜尋樹

```Python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:

        if root is None:
            return True

        s = [(root, float('-inf'), float('inf'))]
        while len(s) > 0:
            node, low, up = s.pop()
            if node.left is not None:
                if node.left.val <= low or node.left.val >= node.val:
                    return False
                s.append((node.left, low, node.val))
            if node.right is not None:
                if node.right.val <= node.val or node.right.val >= up:
                    return False
                s.append((node.right, node.val, up))
        return True
```

### [insert-into-a-binary-search-tree](https://leetcode.com/problems/insert-into-a-binary-search-tree/)

> 給定二元搜尋樹（BST）的根節點和要插入樹中的值，將值插入二元搜尋樹。 返回插入後二元搜尋樹的根節點。 保證原始二元搜尋樹中不存在新值。

```Python
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:

        if root is None:
            return TreeNode(val)

        if val > root.val:
            root.right = self.insertIntoBST(root.right, val)
        else:
            root.left = self.insertIntoBST(root.left, val)

        return root
```

### [delete-node-in-a-bst](https://leetcode.com/problems/delete-node-in-a-bst/)

> 給定一個二元搜尋樹的根節點 root 和一個值 key，刪除二元搜尋樹中的  key  對應的節點，並保證二元搜尋樹的性質不變。返回二元搜尋樹（有可能被更新）的根節點的引用。

```Python
class Solution:
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:

        # try to find the node
        dummy = TreeNode(left=root)
        parent, node = dummy, root
        isleft = True
        while node is not None and node.val != key:
            parent = node
            isleft = key < node.val
            node = node.left if isleft else node.right

        # if found
        if node is not None:
            if node.right is None:
                if isleft:
                    parent.left = node.left
                else:
                    parent.right = node.left
            elif node.left is None:
                if isleft:
                    parent.left = node.right
                else:
                    parent.right = node.right
            else:
                p, n = node, node.left
                while n.right is not None:
                    p, n = n, n.right
                if p != node:
                    p.right = n.left
                else:
                    p.left = n.left
                n.left, n.right = node.left, node.right
                if isleft:
                    parent.left = n
                else:
                    parent.right = n

        return dummy.left
```

### [balanced-binary-tree](https://leetcode.com/problems/balanced-binary-tree/)

> 給定一個二元樹，判斷它是否是高度平衡的二元樹。

```Python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:

        # post-order iterative

        s = [[TreeNode(), -1, -1]]
        node, last = root, None
        while len(s) > 1 or node is not None:
            if node is not None:
                s.append([node, -1, -1])
                node = node.left
                if node is None:
                    s[-1][1] = 0
            else:
                peek = s[-1][0]
                if peek.right is not None and last != peek.right:
                    node = peek.right
                else:
                    if peek.right is None:
                        s[-1][2] = 0
                    last, dl, dr = s.pop()
                    if abs(dl - dr) > 1:
                        return False
                    d = max(dl, dr) + 1
                    if s[-1][1] == -1:
                        s[-1][1] = d
                    else:
                        s[-1][2] = d

        return True
```

### [valid-bfs-of-bst](./bst_bfs.py)

> 給定一個整數數組，求問此數組是不是一個 BST 的 BFS 順序。

此題是麵試真題，但是冇有在 leetcode 上找到原題。由於做法比較有趣也很有 BST 的特點，補充在這供參考。

```Python
import collections

def bst_bfs(A):

    N = len(A)
    interval = collections.deque([(float('-inf'), A[0]), (A[0], float('inf'))])

    for i in range(1, N):
        while interval:
            lower, upper = interval.popleft()
            if lower < A[i] < upper:
                interval.append((lower, A[i]))
                interval.append((A[i], upper))
                break

        if not interval:
            return False

    return True

if __name__ == "__main__":
    A = [10, 8, 11, 1, 9, 0, 5, 3, 6, 4, 12]
    print(bst_bfs(A))
    A = [10, 8, 11, 1, 9, 0, 5, 3, 6, 4, 7]
    print(bst_bfs(A))
```

## 練習

- [ ] [validate-binary-search-tree](https://leetcode.com/problems/validate-binary-search-tree/)
- [ ] [insert-into-a-binary-search-tree](https://leetcode.com/problems/insert-into-a-binary-search-tree/)
- [ ] [delete-node-in-a-bst](https://leetcode.com/problems/delete-node-in-a-bst/)
- [ ] [balanced-binary-tree](https://leetcode.com/problems/balanced-binary-tree/)
