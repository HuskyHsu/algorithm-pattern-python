# 滑動視窗

## 樣板

```cpp
/* 滑動視窗算法框架 */
void slidingWindow(string s, string t) {
    unordered_map<char, int> need, window;
    for (char c : t) need[c]++;

    int left = 0, right = 0;
    int valid = 0;
    while (right < s.size()) {
        // c 是將移入視窗的字符
        char c = s[right];
        // 右移視窗
        right++;
        // 進行視窗內數據的一係列更新
        ...

        /*** debug 輸出的位置 ***/
        printf("window: [%d, %d)\n", left, right);
        /********************/

        // 判斷左側視窗是否要收縮
        while (window needs shrink) {
            // d 是將移出視窗的字符
            char d = s[left];
            // 左移視窗
            left++;
            // 進行視窗內數據的一係列更新
            ...
        }
    }
}
```

需要變化的地方

- 1、右指針右移之後視窗數據更新
- 2、判斷視窗是否要收縮
- 3、左指針右移之後視窗數據更新
- 4、根據題意計算結果

## 示例

### [minimum-window-substring](https://leetcode.com/problems/minimum-window-substring/)

> 給你一個字符串 S、一個字符串 T，請在字符串 S 裏麵找出：包含 T 所有字母的最小子串

```Python
class Solution:
    def minWindow(self, s: str, t: str) -> str:

        target = collections.defaultdict(int)
        window = collections.defaultdict(int)

        for c in t:
            target[c] += 1

        min_size = len(s) + 1
        min_str = ''

        l, r, count, num_char = 0, 0, 0, len(target)

        while r < len(s):
            c = s[r]
            r += 1

            if c in target:
                window[c] += 1

                if window[c] == target[c]:
                    count += 1

                    if count == num_char:
                        while l < r and count == num_char:
                            c = s[l]
                            l += 1

                            if c in target:
                                window[c] -= 1

                                if window[c] == target[c] - 1:
                                    count -= 1

                        if min_size > r - l + 1:
                            min_size = r - l + 1
                            min_str = s[l - 1:r]

        return min_str
```

### [permutation-in-string](https://leetcode.com/problems/permutation-in-string/)

> 給定兩個字符串  **s1**  和  **s2**，寫一個函數來判斷  **s2**  是否包含  **s1 **的排列。

```Python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:

        target = collections.defaultdict(int)

        for c in s1:
            target[c] += 1

        r, num_char = 0, len(target)

        while r < len(s2):
            if s2[r] in target:
                l, count = r, 0
                window = collections.defaultdict(int)
                while r < len(s2):
                    c = s2[r]
                    if c not in target:
                        break
                    window[c] += 1
                    if window[c] == target[c]:
                        count += 1
                        if count == num_char:
                            return True
                    while window[c] > target[c]:
                        window[s2[l]] -= 1
                        if window[s2[l]] == target[s2[l]] - 1:
                            count -= 1
                        l += 1
                    r += 1
            else:
                r += 1

        return False
```

### [find-all-anagrams-in-a-string](https://leetcode.com/problems/find-all-anagrams-in-a-string/)

> 給定一個字符串  **s **和一個非空字符串  **p**，找到  **s **中所有是  **p **的字母異位詞的子串，返回這些子串的起始索引。

```Python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:

        target = collections.defaultdict(int)

        for c in p:
            target[c] += 1

        r, num_char = 0, len(target)

        results = []
        while r < len(s):
            if s[r] in target:
                l, count = r, 0
                window = collections.defaultdict(int)
                while r < len(s):
                    c = s[r]
                    if c not in target:
                        break
                    window[c] += 1
                    if window[c] == target[c]:
                        count += 1
                        if count == num_char:
                            results.append(l)
                            window[s[l]] -= 1
                            count -= 1
                            l += 1
                    while window[c] > target[c]:
                        window[s[l]] -= 1
                        if window[s[l]] == target[s[l]] - 1:
                            count -= 1
                        l += 1
                    r += 1
            else:
                r += 1

        return results
```

### [longest-substring-without-repeating-characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

> 給定一個字符串，請你找出其中不含有重複字符的   最長子串   的長度。
> 示例  1:
>
> 輸入: "abcabcbb"
> 輸出: 3
> 解釋: 因為無重複字符的最長子串是 "abc"，所以其長度為 3。

```Python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:

        last_idx = {}

        l, max_length = 0, 0
        for r, c in enumerate(s):
            if c in last_idx and last_idx[c] >= l:
                max_length = max(max_length, r - l)
                l = last_idx[c] + 1
            last_idx[c] = r

        return max(max_length, len(s) - l) # note that the last substring is not judged in the loop
```

## 總結

- 和雙指針題目類似，更像雙指針的升級版，滑動視窗核心點是維護一個視窗集，根據視窗集來進行處理
- 核心步驟
  - right 右移
  - 收縮
  - left 右移
  - 求結果

## 練習

- [ ] [minimum-window-substring](https://leetcode.com/problems/minimum-window-substring/)
- [ ] [permutation-in-string](https://leetcode.com/problems/permutation-in-string/)
- [ ] [find-all-anagrams-in-a-string](https://leetcode.com/problems/find-all-anagrams-in-a-string/)
- [ ] [longest-substring-without-repeating-characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
