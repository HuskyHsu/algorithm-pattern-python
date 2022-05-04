# 使用 Python3 寫算法題

這裏簡單介紹使用 Python3 寫算法題時的一些特點。本項目並不是一個 Python3 教程，所以預設大家對 Python3 有一定的了解，對於零基礎的同學建議首先了解一下 Python3 的基本文法等基礎知識。

## 邏輯

進行 coding 麵試時，如果不指定使用的編程語言，一般來講考察的是做題的思路而不是編程本身，因此不需要從零開始實現一些基礎的數據結構或算法，利用語言的一些特性和自帶的標準庫可以大大簡化代碼，提高做題速度。下麵會總結一些 Python3 常用的特性，標準算法和數據結構。

## 常用特性

Python 語言有很多特性可以大大簡化代碼，下麵列舉幾個常用的。

#### 數組初始化

```Python
# 初始化一個長度為 N 的一維數組
Array = [0] * N

# 初始化一個形狀為 MxN 的二維數組(矩陣)
Matrix = [[0] * N for _ in range(M)] # 思考：可以寫成 [[0] * N] * M 嗎？
```

#### 交換元素值

```Python
# c語言風格的交換兩個元素值
tmp = a
a = b
b = tmp

# python風格
a, b = b, a
```

#### 連續不等式或等式

```Python
# 判斷 a，b，c 是否相等，Python裏可以直接寫連等
if a == b == c:
    return True

# 不等式也可以
if a <= b < c:
    return True
```

## 標準算法

#### 排序

Python 中排序主要使用 sorted() 和 .sort() 函數，在[官網](https://docs.python.org/3/howto/sorting.html)有詳細介紹，大家可以自行閱讀。

#### 二分查找和插入

Python 自帶的 [bisect](https://docs.python.org/3/library/bisect.html) 庫可以實現二分查找和插入，非常方便。

## 標準數據結構

#### 棧

Python 中的棧使用自帶的 list 類來實現，可參考[官方文檔](https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks)。

#### 隊列

使用 collections 庫中的 deque 類實現，可參考[官方文檔](https://docs.python.org/3/library/collections.html#collections.deque)。

#### 堆

Python 中冇有真的 heap 類，實現堆是使用 list 類配合 heapq 庫中的堆算法，且隻支援最小堆，最大堆需要通過傳入負的優先級來實現，可參考[官方文檔](https://docs.python.org/3.8/library/heapq.html)。

#### HashSet，HashTable

分別通過 [set 類](https://docs.python.org/3.8/library/stdtypes.html#set-types-set-frozenset)和 [dict 類](https://docs.python.org/3/library/stdtypes.html#typesmapping)來實現。

## collections 庫

Python 的 [collections 庫](https://docs.python.org/3/library/collections.html)在刷題時會經常用到，它拓展了一些 Python 中基礎的類，提供了更多功能，例如 defaultdict 可以預設字典中元素 value 的類型，自動提供初始化，Counter 可以直接統計元素出現個數等。

## 總結

以上列舉了一些用 Python3 做算法題時可以用到的一些特性，標準算法和數據結構，總結得肯定不全，因為 Python3 真的有很多可以利用的"騷操作"，大家在學習本項目的時候也會見到，一下記不住也冇關係，多實戰就會了。
