# ALGO  
---  


## BACKTRACKING  
### STRUCTURE CODE  
```javascript
const problem = function() {
  const res = []
  helper()
  return res

  function helper() {
    // end point
    if ()
    else {
      // choose
      do something()
      // explore
      backtracking()
      // un-choose
      back to the condition that do something()
    }
  }
}

```
### EXAMPLES  
#### Permutations  
> Given a collection of distinct integers, return all possible permutations.  
```javascript
Input: [1,2,3]
Output:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```
```javascript
const permute = function(nums) {
  const res = []
  nums.sort((a, b) => a - b) // when the nums may contains duplicated
  helper(nums, [])
  return res
  
  function helper(left, tmp) {
    if (left.length === 0) res.push(tmp.slice())
    else {
      for (let i = 0; i < left.length; i++) {
        // choose
        tmp.push(left[i])
        const del = left.splice(i, 1)[0]
        // explore
        helper(left, tmp)
        // un-choose
        tmp.pop()
        left.splice(i, 0, del)
        // remove duplicate
        while (left[i] === left[i + 1]) {i++}  // when the nums may contains duplicated
      }
    }
  }
}
```
#### Combinations  
> Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.
```javascript
Input: n = 4, k = 2
Output:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
```
```javascript
const combine = function(n, k) {
  const data = [], res = []
  for (let i = 0; i < n; i++) { data[i] = i + 1 }
  helper(0, [])
  return res
  
  function helper(idx, tmp) {
    if (tmp.length === k) res.push(tmp.slice())
    else {
      for (let i = idx; i < n; i++) {
        // choose
        tmp.push(data[i])
        // explore
        helper(i + 1, tmp)
        // un-choose
        tmp.pop()
      }
    }
  }
}
```
> Given a set of candidate numbers (candidates) (without duplicates) and a target number (target),
> find all unique combinations in candidates where the candidate numbers sums to target.  
> The same repeated number may be chosen from candidates unlimited number of times.
```javascript
Input: candidates = [2,3,5], target = 8,
A solution set is:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
```
```javascript
const combinationSum = function(candidates, target) {
  const res = []
  candidates.sort((a, b) => a - b) // when has duplicates
  helper(0, target, [])
  return res
  
  function helper(idx, left, tmp) {
    if (left < 0) return
    else if (left === 0) res.push(tmp.slice())
    else {
      for (let i = idx; i < candidates.length; i++) {
        // choose
        tmp.push(candidates[i])
        // explore
        helper(i, left - candidates[i], tmp)
        // un-choose
        tmp.pop()
        // remove duplicates
        while (candidates[i] === candidates[i + 1]) {i++} // when has duplicates
      }
    }
  }
}
```
> Find all possible combinations of k numbers that add up to a number n, given that only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.
```javascript
Input: k = 3, n = 9
Output: [[1,2,6], [1,3,5], [2,3,4]]
```
```javascript
const combinationSum3 = function(k, n) {
  const res = [], data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
  helper(n, [], 0)
  return res
  
  function helper(left, tmp, idx) {
    if (left === 0 && tmp.length === k) res.push(tmp.slice())
    else {
      for (let i = idx; i < data.length; i++) {
        // choose
        tmp.push(data[i])
        // explore
        helper(left - data[i], tmp, i + 1)
        // un-choose
        tmp.pop()
      }
    }
  }
}
```
#### N-QUEENS
> As the N-QUEENS description
```javascript
const solveNQueens = function(n) {
  const board = [...Array(n)].map(_ => '.'.repeat(n)), res = []
  helper(board, 0)
  return res
  
  function helper(board, row) {
    if (row === n) res.push(board.slice())
    else {
      for (let col = 0; col < n; col++) {
        // validation
        if (!isValid(row, col, board)) continue
        // choose
        const tmp = board[row].split('')
        tmp[col] = 'Q'
        board[row] = tmp.join('')
        // explore
        helper(board, row + 1)
        // un-choose
        tmp[col] = '.'
        board[row] = tmp.join('')
      }
    }
  }
  
  function isValid(row, col, board) {
    // validate col
    for (let i = 0; i < row; i++) {
      if (board[i][col] === 'Q') return false
    }
    // validate left-top
    for (let i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
      if (board[i][j] === 'Q') return false
    }
    // validate right-top
    for (let i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
      if (board[i][j] === 'Q') return false
    }
    return true
  }
}
```  
> If just count the solutions
```javascript
const totalNQueens = function(n) {
  let res = 0
  const cols = Array(n).fill(false), diag = Array(2n).fill(false), antiDiag = Array(2n).fill(false)
  helper(0)
  return res
  
  function helper(row) {
    if (row === n) res++
    else {
      for (let col = 0; col < n; col++) {
        // validate
        if (!isValid(row, col)) continue
        // choose
        cols[col] = diag[row - col + n] = antiDiag[row + col] = true
        // explore
        helper(row + 1)
        cols[col] = diag[row - col + n] = antiDiag[row + col] = false
        
      }
    }
  }
  
  function isValid(row, col) {
    if (cols[col] || diag[row - col + n] || antiDiag[row + col]) return false
    return true
  }
}
```

#### Generate Parentheses
> Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
```javascript
input: n = 3
output: [
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
```  
```javascript
const generateParenthesis = function(n) {
  const res = []
  helper(0, 0, 0, '')
  return res
  
  function helper(left, right, level, tmp) {
    if (right > left || level > 2 * n) return
    else if (level === 2 * n && left === right) res.push(tmp.slice())
    else {
      helper(left + 1, right, level + 1, tmp + '(')
      helper(left, right + 1, level + 1, tmp + ')')
    }
  }
}
```
### TIME-COMPLEXITY  
According to the specific problems. O(N!)/ O(2^N) / balabala



# BASIC  
---


## NETWORK  
- **网络结构分层**  
 ***OSI 7层：*** 应用层，会话层，表示层，传输层，网络层，数据链路层，物理层  
 ***TCP/IP 4层:***  应用层，传输层，网络层，网络接口层  
 ***综合 5层：*** 应用层，传输层，网络层，数据链路层，物理层  
- **每层的作用和常用协议**  
 *应用层：* 通过应用进程之间的交互完成特定网络应用，该层协议定义应用进程之间的通信和交互规则
 常用协议有：  
 **域名系统`DNS`**, **支持网络的`HTTP`**, **支持电子邮件的`SMTP`**等  

 *传输层：* 负责向两台主机进程之间的通信提供通用的数据传输服务  
 常用的协议有：  
 - **传输控制协议`TCP`**，它提供面向连接的、可靠的数据传输服务，传输单位是`报文段(segment)`  
 - **用户数据报协议`UDP`**，它提供无连接的传输服务，传输单位是`用户数据报`  

 *网络层：* 负责为分组交换网络上不同的主机提供通信服务，在发送数据时网络层吧运输层产生的报文段或用户数据报封装成分组或
 包进行传送。另一个任务是选择合适的路由，使源主机传输层传过来的分组能过通过网络中的路由器找到目的主机  
 常用的协议有：  
 - **网际协议`IP`**,用来使互联起来的计算机网络能够相互通信，（网际层由来）`IPV4`, `IPV6`  
 - **地址解析协议`ARP`**, `IP`使用`ARP协议`，其作用是通过ARP高速缓存存储本地局域网的各主机和路由器的`IP地址`到`MAC地址`的映射表，
 以从网络层的IP地址解析出在数据链路层使用的MAC地址。`RARP`逆地址解析协议，使硬件地址的主机能够找出IP地址，被`DHCP取代`  
 - **网际控制报文协议`ICMP`**, `ICMP报文`作为IP数据报的数据，加上首部后组成`IP数据报`发送出去，ICMP允许主机或者路由器报告差错情况
 和提供有关异常情况的报告。ICMP有两种报文，`差错报告报文`和`询问报文`。其最重要的应用就是`ping`，来测试两台主机之间的连通性，ping使用了
 `ICMP回送请求`与`回送回答报文`  
 - **网际组管理协议`IGMP`**，是IP多播使用的协议，作用是让连接在本地局域网上的多播路由器知道本局域网上是否有某个进程参加或退出了某个多播组
- **TCP & UDP**  
 ***用户数据报协议UDP：***  
 - UDP只在IP的数据报服务上增加了很少一点功能，就是复用和分用以及差错监测  
 其特点主要是：  
 - UDP是无连接的，发送数据前不需要建立连接；
 - UDP使用最大努力交付，不保证数据传输的可靠性；  
 - UDP是面向报文的，发送方UDP对应用程序交下来的报文在添加首部后就向下交付IP层；  
 - UDP没有拥塞控制；UDP支持一对一，一对多，多对一和多对多的交互通信；UDP首部开销小，只有8个字节，TCP需要20个字节  

 ***传输控制协议TCP：***  
 其主要特点是：  
 - TCP是面向连接的运输层协议，即TCP在进行数据通信前需要建立连接，主要是通过三次握手机制实现，在进行数据通信后，需要断开连接，通过四次挥手机制实现  
 - 每条TCP连接只能有两个端点  
 - TCP提供可靠的交付服务，通过TCP发送的数据无差错，不丢失，不重复  
 - TCP是全双工通信，在发送端和接收端没有缓存，发送发将数据发送到缓存后，接收方将数据放入缓存，上层应用程序会在合适时机获取数据  
 - TCP是面向字节流的，所谓流就是流入进程或者进程重流出的字节序列。虽然应用进程和TCP交互是一次一个数据块，但是TCP会将数据块看成  
 一连串无结构的字节流，不能保证发送的数据块和接收的数据块大小一致，但是字节流是完全一样的  
 - **Q：如何保证可靠？**  
 TCP的发送的报文是交给IP层传送的，而IP只能提供尽最大努力服务，所以TCP必须采取适当的措施才能使得两个运输层之间的通信变得可靠。  
 理想的通信有两个条件，第一是传输的数据不会出现差错，第二是无论发生数据的速度有多快，接收端都来得及接收。但是在现实的网络环境下  
 几乎是不可能实现的，TCP使用了重传机制来解决传输数据出错的问题，使用流量控制来降低发送端的速度，以便接收端来的及接收
- **停止等待协议**  
 每发送一个分组就停下来，等收到了对方对该分组的确认之后再继续发送下一个分组。每发送完一个分组就设置一个超时计时器，  
 如果在规定的时间内没有收到分组的确认消息，就会进行超时重传。在规定时间内收到了确认消息就会撤销计时器
 同时需要注意三点：  
 1.计时器设置的超时时间应该稍微长于分组的往返时间，如果时间太长通信效率就会很低，如果时间过短会产生不必要的重传，浪费网络资源  
 2.每一个分组都设有一个副本，以便超时重传时使用，当收到了分组的确认后再进行清除  
 3.分组和确认分组都必须进行编号，这样才能明确是哪一个分组收到了确认  
- **ARQ协议/重传机制**  
 假设分组的包确认丢失了，发送方在设定的超时时间内没有收到确认，不知道是自己发送的分组丢失还是接收方的确认丢失，  
 因此发送方需要重传分组。当接收方收到了分组后就丢失这个分组，重新发送确认  
 还有一种情况是分组没有丢失但是晚到了，发送端会受到重复确认，接收端仍然会收到重复的分组，同样丢弃并确认  
 上述确认和重传机制，即`ARQ(Automatic Repeat reQuest)`，自动重传请求，接收端不需要向发送端发送重传请求，当超过指定时间时发送端会自动进行超时重传  
- **效率问题**  
 停止等待协议的优点是简单，缺点是信道利用率太低。信道利用率为TD/(TD+RTT+TA)，TD是发送分组的时间，T2是发送确认分组的时间，RTT是往返时间，  
 当RTT远大于TD时通信效率就会非常低。为了提高传输效率，可以采用流水线传输，例如连续ARQ协议和滑动窗口机制  
- **连续ARQ**  
 连续ARQ规定每收到一个确认就把发送窗口向前滑动一个分组的位置，接收方一般采用累积确认的方式，就是说接收方不必对收到的分组逐个确认，  
 只需要对按序到达的最后一个分组进行确认。优点是实现容易，即使确认丢失也不必重传，缺点是不能向发送方反映出接收方已经正确收到的所有分组的消息。  
 例如发送方发送了5个分组，第3个分组丢失了，接收方只能确认前2个，发送方必须把后面3个都重新发送
- **滑动窗口机制**  
 滑动窗口以字节为单位。发送端有一个发送窗口，窗口中的序号是允许发送的序号，窗口的后沿是已经发送并且确认的序号，窗口的前沿是不允许发送的序号。  
 窗口的后沿可能不动（代表没有收到新的确认），也有可能前移（代表收到了新的确认），但是不会后移（不可能撤销已经确认的数据）。  
 窗口的前沿一般是向前的，也有可能不动（表示没有收到新的请求或对方的接收窗口变小），也有可能收缩，但是TCP强烈不建议这么做，  
 因为发送端在收到通知前可能已经发送了很多数据，此时如果收缩窗口可能会产生错误
- **tcp 三次握手 四次挥手**  


- **网络拥塞控制四种算法，慢开始，拥塞避免，快重传，快恢复**

- **tcp连接和释放中的状态有哪些，以及如果日志中出现某些状态码过多如何处理**

- **http连接中状态码有哪些，如果出现某些错误的状态码，分析出是什么情况吗**

- **http2.0和http1.1的区别，http2.0的原理**

- **GET 和 POST**  