# GO 快速入門

## 基本文法

[Go 語言聖經](https://books.studygolang.com/gopl-zh/)

## 常用庫

### 切片

go 通過切片類比堆疊與佇列

堆疊

```go
// 創建堆疊
stack:=make([]int,0)
// push壓入
stack=append(stack,10)
// pop彈出
v:=stack[len(stack)-1]
stack=stack[:len(stack)-1]
// 檢查堆疊空
len(stack)==0
```

佇列

```go
// 創建佇列
queue:=make([]int,0)
// enqueue入隊
queue=append(queue,10)
// dequeue出隊
v:=queue[0]
queue=queue[1:]
// 長度0為空
len(queue)==0
```

注意點

- 參數傳遞，隻能修改，不能新增或者刪除原始數據
- 預設 s=s[0:len(s)]，取下限不取上限，數學錶示為：[)

### 字典

基本用法

```go
// 創建
m:=make(map[string]int)
// 設定kv
m["hello"]=1
// 刪除k
delete(m,"hello")
// 遍歷
for k,v:=range m{
    println(k,v)
}
```

注意點

- map 鍵需要可比較，不能為 slice、map、function
- map 值都有預設值，可以直接操作預設值，如：m[age]++ 值由 0 變為 1
- 比較兩個 map 需要遍歷，其中的 kv 是否相同，因為有預設值關係，所以需要檢查 val 和 ok 兩個值

### 標準庫

sort

```go
// int排序
sort.Ints([]int{})
// 字符串排序
sort.Strings([]string{})
// 自定義排序
sort.Slice(s,func(i,j int)bool{return s[i]<s[j]})
```

math

```go
// int32 最大最小值
math.MaxInt32 // 實際值：1<<31-1
math.MinInt32 // 實際值：-1<<31
// int64 最大最小值（int預設是int64）
math.MaxInt64
math.MinInt64

```

copy

```go
// 刪除a[i]，可以用 copy 將i+1到末尾的值覆蓋到i,然後末尾-1
copy(a[i:],a[i+1:])
a=a[:len(a)-1]

// make創建長度，則通過索引賦值
a:=make([]int,n)
a[n]=x
// make長度為0，則通過append()賦值
a:=make([]int,0)
a=append(a,x)
```

### 常用技巧

類型轉換

```go
// byte轉數字
s="12345"  // s[0] 類型是byte
num:=int(s[0]-'0') // 1
str:=string(s[0]) // "1"
b:=byte(num+'0') // '1'
fmt.Printf("%d%s%c\n", num, str, b) // 111

// 字符串轉數字
num,_:=strconv.Atoi()
str:=strconv.Itoa()

```

## 刷題注意點

- leetcode 中，全局變數不要當做返回值，否則刷題檢查器會報錯
