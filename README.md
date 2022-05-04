# 說明

本項目為原項目 [algorithm-pattern](https://github.com/greyireland/algorithm-pattern) 的 Python3 語言實現版本，原項目使用 go 語言實現，目前已獲 ![GitHub stars](https://img.shields.io/github/stars/greyireland/algorithm-pattern?style=social)。在原項目基礎上，本項目添加了優先級佇列，併查集，圖相關算法等內容，基本覆蓋了所有基礎數據結構和算法，非常適合找工刷題的同學快速上手。以下為原項目 README，目錄部分增加了本項目的新內容。

# 算法模板

算法模板，最科學的刷題方式，最快速的刷題路徑，一個月從入門到 offer，你值得擁有 🐶~

算法模板顧名思義就是刷題的套路模板，掌握了刷題模板之後，刷題也變得好玩起來了~

> 此項目是自己找工作時，從 0 開始刷 LeetCode 的心得記錄，通過各種刷題文章、專欄、視訊等總結了一套自己的刷題模板。
>
> 這個模板主要是介紹了一些通用的刷題模板，以及一些常見問題，如到底要刷多少題，按什麼順序來刷題，如何提高刷題效率等。

## 在線文檔

在線文檔 Gitbook：[算法模板 🔥](https://greyireland.gitbook.io/algorithm-pattern/)

## 核心內容

### 入門篇 🐶

- [使用 Python3 寫算法題](./introduction/python.md)
- [算法快速入門](./introduction/quickstart.md)

### 數據結構篇 🐰

- [二元樹](./data_structure/binary_tree.md)
- [鏈結串列](./data_structure/linked_list.md)
- [堆疊與佇列](./data_structure/stack_queue.md)
- [堆積](./data_structure/heap.md)
- [併查集](./data_structure/union_find.md)
- [二進位](./data_structure/binary_op.md)

### 基礎算法篇 🐮

- [二分搜尋](./basic_algorithm/binary_search.md)
- [排序算法](./basic_algorithm/sort.md)
- [動態規劃](./basic_algorithm/dp.md)
- [圖相關算法](./basic_algorithm/graph/)

### 算法思維 🦁

- [遞迴思維](./advanced_algorithm/recursion.md)
- [滑動視窗思想](./advanced_algorithm/slide_window.md)
- [二元搜尋樹](./advanced_algorithm/binary_search_tree.md)
- [回溯法](./advanced_algorithm/backtrack.md)

## 心得體會

文章大部分是對題目的思路介紹，和一些問題的解析，有了思路還是需要自己手動寫寫的，所以每篇文章最後都有對應的練習題

刷完這些練習題，基本對數據結構和算法有自己的認識體會，基本大部分麵試題都能寫得出來，國內的 BAT、TMD 應該都不是問題

從 4 月份找工作開始，從 0 開始刷 LeetCode，中間大概花了一個半月(6 周)左右時間刷完 240 題。

![一個半月刷完240題](https://img.fuiboom.com/img/leetcode_time.png)

![刷題記錄](https://img.fuiboom.com/img/leetcode_record.png)

開始刷題時，確實是無從下手，因為從序號開始刷，刷到幾道題就遇到 hard 的題型，會卡住很久，後麵去評論區看別人怎麼刷題，也去 Google 搜索最好的刷題方式，發現按題型刷題會舒服很多，基本一個類型的題目，一天能做很多，慢慢刷題也不再枯燥，做起來也很有意思，最後也收到不錯的 offer（最後去了宇宙係）。

回到最開始的問題，麵試到底要刷多少題，其實這個取決於你想進什麼樣公司，你定的目標如果是國內一線大廠，個人感覺大概 200 至 300 題基本就滿足大部分麵試需要了。第二個問題是按什麼順序刷及如何提高效率，這個也是本 repo 的目的，給你指定了一個刷題的順序，以及刷題的模板，有了方嚮和技巧後，就去動手吧~ 希望刷完之後，你也能自己總結一套屬於自己的刷題模板，有所收獲，有所成長~

## 推薦的刷題路徑

按此 repo 目錄刷一遍，如果中間有題目卡住了先跳過，然後刷題一遍 LeetCode 探索基礎卡片，最後快要麵試時刷題一遍劍指 offer。

為什麼這麼要這麼刷，因為 repo 裏麵的題目是按類型歸類，都是一些常見的高頻題，很有代錶性，大部分都是可以用模板加一點變形做出來，刷完後對大部分題目有基本的認識。然後刷一遍探索卡片，鞏固一下一些基礎知識點，總結這些知識點。最後劍指 offer 是大部分公司的出題源頭，刷完麵試中基本會遇到現題或者變形題，基本刷完這三部分，大部分國內公司的麵試題應該就冇什麼問題了~

1、 [algorithm-pattern 練習題](https://greyireland.gitbook.io/algorithm-pattern/)

![練習題](https://img.fuiboom.com/img/repo_practice.png)

2、 [LeetCode 卡片](https://leetcode.com/explore/)

![探索卡片](https://img.fuiboom.com/img/leetcode_explore.png)

3、 [劍指 offer](https://leetcode.com/problemset/lcof/)

![劍指offer](https://img.fuiboom.com/img/leetcode_jzoffer.png)

刷題時間可以合理分配，如果打算準備麵試了，建議前麵兩部分 一個半月 （6 周）時間刷完，最後劍指 offer 半個月刷完，邊刷可以邊投簡曆進行麵試，遇到不會的不用著急，往模板上套就對了，如果麵試官給你提示，那就好好做，不要錯過這大好機會~

> 注意點：如果為了找工作刷題，遇到 hard 的題如果有思路就做，冇思路先跳過，先把基礎打好，再來刷 hard 可能效果會更好~

## 麵試資源

分享一些計算機的經典書籍，大部分對麵試應該都有幫助，強烈推薦 🌝

[我看過的 100 本書](https://github.com/greyireland/awesome-programming-books-1)

## 後續

持續更新中，覺得還可以的話點個 **star** 收藏呀 ⭐️~

【 Github 】[https://github.com/greyireland/algorithm-pattern](https://github.com/greyireland/algorithm-pattern) ⭐️
