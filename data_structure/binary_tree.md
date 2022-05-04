# 二元樹

## 知識點

### 二元樹遍曆

**前序遍曆**：**先訪問根節點**，再前序遍曆左子樹，再前序遍曆右子樹
**中序遍曆**：先中序遍曆左子樹，**再訪問根節點**，再中序遍曆右子樹
**後序遍曆**：先後序遍曆左子樹，再後序遍曆右子樹，**再訪問根節點**

註意點

- 以根訪問順序決定是什麼遍曆
- 左子樹都是優先右子樹

#### 遞迴模闆

- 遞迴實現二元樹遍曆非常簡單，不同順序區別僅在於訪問父結點順序

```Python
def preorder_rec(root):
    if root is None:
        return
    visit(root)
    preorder_rec(root.left)
    preorder_rec(root.right)
    return

def inorder_rec(root):
    if root is None:
        return
    inorder_rec(root.left)
    visit(root)
    inorder_rec(root.right)
    return

def postorder_rec(root):
    if root is None:
        return
    postorder_rec(root.left)
    postorder_rec(root.right)
    visit(root)
    return
```

#### [前序非遞迴](https://leetcode.com/problems/binary-tree-preorder-traversal/)

- 本質上是圖的 DFS 的一個特例，因此可以用棧來實現

```Python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:

        preorder = []
        if root is None:
            return preorder

        s = [root]
        while len(s) > 0:
            node = s.pop()
            preorder.append(node.val)
            if node.right is not None:
                s.append(node.right)
            if node.left is not None:
                s.append(node.left)

        return preorder
```

#### [中序非遞迴](https://leetcode.com/problems/binary-tree-inorder-traversal/)

```Python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        s, inorder = [], []
        node = root
        while len(s) > 0 or node is not None:
            if node is not None:
                s.append(node)
                node = node.left
            else:
                node = s.pop()
                inorder.append(node.val)
                node = node.right
        return inorder
```

#### [後序非遞迴](https://leetcode.com/problems/binary-tree-postorder-traversal/)

```Python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:

        s, postorder = [], []
        node, last_visit = root, None

        while len(s) > 0 or node is not None:
            if node is not None:
                s.append(node)
                node = node.left
            else:
                peek = s[-1]
                if peek.right is not None and last_visit != peek.right:
                    node = peek.right
                else:
                    last_visit = s.pop()
                    postorder.append(last_visit.val)


        return postorder
```

註意點

- 核心就是：根節點必須在右節點彈出之後，再彈出

DFS 深度搜索-從下嚮上（分治法）

```Python
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:

        if root is None:
            return []

        left_result = self.preorderTraversal(root.left)
        right_result = self.preorderTraversal(root.right)

        return [root.val] + left_result + right_result
```

註意點：

> DFS 深度搜索（從上到下） 和分治法區別：前者一般將最終結果通過指針參數傳入，後者一般遞迴返回結果最後合並

#### [BFS 層次遍曆](https://leetcode.com/problems/binary-tree-level-order-traversal/)

```Python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:

        levels = []
        if root is None:
            return levels

        bfs = collections.deque([root])

        while len(bfs) > 0:
            levels.append([])

            level_size = len(bfs)
            for _ in range(level_size):
                node = bfs.popleft()
                levels[-1].append(node.val)

                if node.left is not None:
                    bfs.append(node.left)
                if node.right is not None:
                    bfs.append(node.right)

        return levels
```

### 分治法應用

先分別處理局部，再合並結果

適用場景

- 快速排序
- 歸並排序
- 二元樹相關問題

分治法模闆

- 遞迴返回條件
- 分段處理
- 合並結果

## 常見題目示例

### [maximum-depth-of-binary-tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

> 給定一個二元樹，找出其最大深度。

- 思路 1：分治法

```Python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:

        if root is None:
            return 0

        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```

- 思路 2：層序遍曆

```Python
class Solution:
    def maxDepth(self, root: TreeNode) -> List[List[int]]:

        depth = 0
        if root is None:
            return depth

        bfs = collections.deque([root])

        while len(bfs) > 0:
            depth += 1
            level_size = len(bfs)
            for _ in range(level_size):
                node = bfs.popleft()
                if node.left is not None:
                    bfs.append(node.left)
                if node.right is not None:
                    bfs.append(node.right)

        return depth
```

### [balanced-binary-tree](https://leetcode.com/problems/balanced-binary-tree/)

> 給定一個二元樹，判斷它是否是高度平衡的二元樹。

- 思路 1：分治法，左邊平衡 && 右邊平衡 && 左右兩邊高度 <= 1，

```Python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:

        def depth(root):

            if root is None:
                return 0, True

            dl, bl = depth(root.left)
            dr, br = depth(root.right)

            return max(dl, dr) + 1, bl and br and abs(dl - dr) < 2

        _, out = depth(root)

        return out
```

- 思路 2：使用後序遍曆實現分治法的叠代版本

```Python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:

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

### [binary-tree-maximum-path-sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

> 給定一個**非空**二元樹，返回其最大路徑和。

- 思路：分治法。最大路徑的可能情況：左子樹的最大路徑，右子樹的最大路徑，或通過根結點的最大路徑。其中通過根結點的最大路徑值等於以左子樹根結點為端點的最大路徑值加以右子樹根結點為端點的最大路徑值再加上根結點值，這裏還要考慮有負值的情況即負值路徑需要丟棄不取。

```Python
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:

        self.maxPath = float('-inf')

        def largest_path_ends_at(node):
            if node is None:
                return float('-inf')

            e_l = largest_path_ends_at(node.left)
            e_r = largest_path_ends_at(node.right)

            self.maxPath = max(self.maxPath, node.val + max(0, e_l) + max(0, e_r), e_l, e_r)

            return node.val + max(e_l, e_r, 0)

        largest_path_ends_at(root)
        return self.maxPath
```

### [lowest-common-ancestor-of-a-binary-tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)

> 給定一個二元樹, 找到該樹中兩個指定節點的最近公共祖先。

- 思路：分治法，有左子樹的公共祖先或者有右子樹的公共祖先，就返回子樹的祖先，否則返回根節點

```Python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

        if root is None:
            return None

        if root == p or root == q:
            return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left is not None and right is not None:
            return root
        elif left is not None:
            return left
        elif right is not None:
            return right
        else:
            return None
```

### BFS 層次應用

### [binary-tree-zigzag-level-order-traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)

> 給定一個二元樹，返回其節點值的鋸齒形層次遍曆。Z 字形遍曆

- 思路：在 BFS 叠代模闆上改用雙端隊列控製輸出順序

```Python
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:

        levels = []
        if root is None:
            return levels

        s = collections.deque([root])

        start_from_left = True
        while len(s) > 0:
            levels.append([])
            level_size = len(s)

            if start_from_left:
                for _ in range(level_size):
                    node = s.popleft()
                    levels[-1].append(node.val)
                    if node.left is not None:
                        s.append(node.left)
                    if node.right is not None:
                        s.append(node.right)
            else:
                for _ in range(level_size):
                    node = s.pop()
                    levels[-1].append(node.val)
                    if node.right is not None:
                        s.appendleft(node.right)
                    if node.left is not None:
                        s.appendleft(node.left)

            start_from_left = not start_from_left


        return levels
```

### 二元搜尋樹應用

### [validate-binary-search-tree](https://leetcode.com/problems/validate-binary-search-tree/)

> 給定一個二元樹，判斷其是否是一個有效的二元搜尋樹。

- 思路 1：中序遍曆後檢查輸出是否有序，缺點是如果不平衡無法提前返回結果， 代碼略

- 思路 2：分治法，一個二元樹為合法的二元搜尋樹當且僅當左右子樹為合法二元搜尋樹且根結點值大於右子樹最小值小於左子樹最大值。缺點是若不用叠代形式實現則無法提前返回，而叠代實現右比較複雜。

```Python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:

        if root is None: return True

        def valid_min_max(node):

            isValid = True
            if node.left is not None:
                l_isValid, l_min, l_max = valid_min_max(node.left)
                isValid = isValid and node.val > l_max
            else:
                l_isValid, l_min = True, node.val

            if node.right is not None:
                r_isValid, r_min, r_max = valid_min_max(node.right)
                isValid = isValid and node.val < r_min
            else:
                r_isValid, r_max = True, node.val


            return l_isValid and r_isValid and isValid, l_min, r_max

        return valid_min_max(root)[0]
```

- 思路 3：利用二元搜尋樹的性質，根結點為左子樹的右邊界，右子樹的左邊界，使用先序遍曆自頂嚮下更新左右子樹的邊界並檢查是否合法，叠代版本實現簡單且可以提前返回結果。

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

#### [insert-into-a-binary-search-tree](https://leetcode.com/problems/insert-into-a-binary-search-tree/)

> 給定二元搜尋樹（BST）的根節點和要插入樹中的值，將值插入二元搜尋樹。 返回插入後二元搜尋樹的根節點。

- 思路：如果隻是為了完成任務則找到最後一個葉子節點滿足插入條件即可。但此題深挖可以涉及到如何插入並維持平衡二元搜尋樹的問題，並不適合初學者。

```Python
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:

        if root is None:
            return TreeNode(val)

        node = root
        while True:
            if val > node.val:
                if node.right is None:
                    node.right = TreeNode(val)
                    return root
                else:
                    node = node.right
            else:
                if node.left is None:
                    node.left = TreeNode(val)
                    return root
                else:
                    node = node.left
```

## 總結

- 掌握二元樹遞迴與非遞迴遍曆
- 理解 DFS 前序遍曆與分治法
- 理解 BFS 層次遍曆

## 練習

- [ ] [maximum-depth-of-binary-tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
- [ ] [balanced-binary-tree](https://leetcode.com/problems/balanced-binary-tree/)
- [ ] [binary-tree-maximum-path-sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)
- [ ] [lowest-common-ancestor-of-a-binary-tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)
- [ ] [binary-tree-level-order-traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
- [ ] [binary-tree-level-order-traversal-ii](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/)
- [ ] [binary-tree-zigzag-level-order-traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)
- [ ] [validate-binary-search-tree](https://leetcode.com/problems/validate-binary-search-tree/)
- [ ] [insert-into-a-binary-search-tree](https://leetcode.com/problems/insert-into-a-binary-search-tree/)
