# Hot Algo  

## 2. 2 Sum  
[link]()  
```javascript
const addTwoNumbers = function(l1, l2) {
  let cur = new ListNode(0)
  let l0 = cur
  let carry = 0
  
  while (l1 || l2 || carry) {
    let v1 = l1 ? l1.val : 0
    let v2 = l2 ? l2.val : 0
    cur.next = new ListNode((v1 + v2 + carry) % 10)
    cur = cur.next
    carry = (v1 + v2 + carry) / 10 | 0
    if (l1) l1 = l1.next
    if (l2) l2 = l2.next
  }
  return l0.next
}
```
## 3. Longest Substring Without Repeating Characters  
[link](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
```javascript
const lengthOfLongestSubstring = function(s) {
  let map = new Map()
  let head = 0, res = 0
  for (let i = 0; i < s.length; i++) {
    if (map.has(s[i]) && map.get(s[i]) >= head) head = map.get(s[i]) + 1
    map.set(s[i], i)
    res = Math.max(res, i - head + 1)
  }
  return res
}
```
## 4. Median Of Two Sorted Arrays  
[link](https://leetcode.com/problems/median-of-two-sorted-arrays/)  
```javascript
const findMedian = function(nums1, nums2) {
  let len1 = nums1.length, len2 = nums2.length
  let resLeft = 0, resRight = 0
  if (len1 > len2) {
    let tmp = len2, tmpNums = nums2
    len2 = len1
    nums2 = nums1
    len1 = tmp
    nums1 = tmpNums
  }
  let lo = 0, hi = len1, halfLen = (len1 + len2 + 1) >> 1
  while (lo <= hi) {
    let i = lo + ((hi - lo) >> 1), j = halfLen - i
    if (i < len1 && nums2[j - 1] > nums1[i]) lo = i + 1
    else if (i > 0 && nums1[i - 1] > nums2[j]) hi = i - 1
    else {
      if (i === 0) resLeft = nums2[j - 1]
      else if (j === 0) resLeft = nums1[i - 1]
      else resLeft = Math.max(nums1[i - 1], nums2[j - 1])

      if ((len1 + len2) & 1) return resLeft
      if (i === m) resRight = nums2[j]
      else if (j === n) resRight = nums1[i]
      else resRight = Math.min(num1[i], nums2[j])
      return (resLeft + resRight) / 2.0
    }
  }
}
```

## 15. 3Sum  
[link](https://leetcode.com/problems/3sum/)  
```javascript
const threeSum = function(nums) {
  let arr = []
  nums.sort((a, b) => a - b)
  for (let i = 0; i < nums.length - 2; i++) {
    if (nums.length < 3) return arr
    if (nums[i] > 0) return arr
    if (i > 0 && nums[i] === nums[i - 1]) continue
    for (let j = i + 1, k = nums.length - 1; j < k;) {
      if (nums[i] + nums[j] + nums[k] === 0) {
        arr.push([nums[i], nums[j], nums[k]])
        j++
        k--
        while (nums[j] === nums[j - 1]) j++
        while (nums[k] === nums[k + 1]) k--
      } else if (nums[i] + nums[j] + nums[k] > 0) k--
      else j++
    }
  }
  return arr
}
```
## 22. Generate Parentheses  
[link](https://leetcode.com/problems/generate-parentheses/)  
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
## 23. Merge k Sorted Lists  
[link](https://leetcode.com/problems/merge-k-sorted-lists/)  
```javascript
const mergeKLists = function(lists) {
  function mergeLists(a, b) {
    const dummy = new ListNode(0)
    let tmp = dummy
    while (a && b) {
      if (a.val < b.val) {
        tmp.next = a
        a = a.next
      } else {
        tmp.next = b
        b = b.next
      }
      tmp = tmp.next
    }
    if (a) tmp.next = a
    if (b) tmp.next = b
    return dummy.next
  }
    
  if (lists.length === 0) return null
  while (lists.length > 1) {
    let a = lists.shift()
    let b = lists.shift()
    const h = mergeLists(a, b)
    lists.push(h)
  }
  return lists[0]
}

```
## 25. Reverse Nodes in k-Group  
[link](https://leetcode.com/problems/reverse-nodes-in-k-group/)  
```javascript
const reverseKGroup = function(head, k) {
  let dummy = new ListNode(0)
  dummy.next = head
  let h = dummy
  while (h) {
    let node = h
    for (let i = 0; i < k && node; i++) {
      node = node.next
    }
    if (node == null) break
    
    let prev = null, cur = h.next, next = null
    for (let i = 0; i < k; i++) {
      next = cur.next
      cur.next = prev
      prev = cur
      cur = next
    }
    let tail = h.next
    tail.next = cur
    h.next = prev
    h = tail
  }
  return dummy.next
}
```
## 33. Search in Rotated Sorted Array  
[link](https://leetcode.com/problems/search-in-rotated-sorted-array/)  
```javascript
const search = function(nums, target) {
  let lo = 0, hi = nums.length - 1
  while (lo <= hi) {
    let mid = lo + ((hi - lo) >> 1)
    if (target === nums[mid]) return mid
    if (nums[mid] > nums[hi]) {
      if (target >= nums[lo] && target < nums[mid]) hi = mid - 1
      else lo = mid + 1
    } else {
      if (target > nums[mid] && target <= nums[hi]) lo = mid + 1
      else hi = mid - 1
    }
  }
  return -1
}
```
## 41. First Missing Positive  
[link](https://leetcode.com/problems/first-missing-positive/)  
```javascript
const irstMissingPositive = function(nums) {
  // put each num in right place
  // missing num in range 1 ~ nums.length
  for (let i = 0; i < nums.length; i++) {
    while (nums[i] > 0 && nums[i] <= nums.length && nums[i] != nums[nums[i] - 1]) {
      swap(i, nums[i] - 1)
    }
  }
  for (let i = 0; i < nums.length; i++) {
    if (nums[i] != i + 1) return i + 1
  }
  return nums.length + 1
}
```
## 42. Trapping Rain Water  //TODO
[link](https://leetcode.com/problems/trapping-rain-water/)
```javascript
const trap = function(height) {
  let l = 0, r = height.length - 1
  let res = 0
  let maxL = 0, maxR = 0
  while (l <= r) {
    if (height[l] <= height[r]) {
      if (height[l] >= maxL) maxL = height[l]
      else res += maxL - height[l]
      l++
    } else {
      if (height[r] >= maxR) maxR = height[r]
      eles res += maxR - height[r]
      r--
    }
  }
  return res
}
```
## 54. Spiral Matrix  
[link](https://leetcode.com/problems/spiral-matrix/)
```javascript
const spiralOrder = function(matrix) {
  if (!matrix || !matrix.length || !matrix[0].length) return []
  let m = matrix.length, n = matrix[0].length
  const res = []
  let left = 0, top = 0, right = n - 1, bot = m - 1
  while (top <= bot && left <= right) {
    // from left to right
    for (let i = left; i <= right; i++) res.push(matrix[top][i])
    top++
    // from top to bot
    for (let i = top; i <= bot; i++) res.push(matrix[i][right])
    right--
    // from right to left
    for (let i = right; i >= left && top <= bot; i--) res.push(matrix[bot][i])
    bot--
    // from bot to top
    for (let i = bot; i >= top && left <= right; i--) res.push(matrix[i][left])
    left++
  }
  return res
}
```
## 56. Merge Intervals  
[link](https://leetcode.com/problems/merge-intervals/)  
```javascript
const merge = function(intervals) {
  if (!intervals.length) return intervals
  intervals.sort((a, b) => a[0] === b[0] ? a[1] - b[1] : a[0] - b[0])
  let prev = intervals[0]
  let res = [prev]
  for (let cur of intervals) {
    if (cur[0] <= prev[1]) prev[1] = Math.max(prev[1], cur[1])
    else {
      res.push(cur)
      prev = cur
    }
  }
  return res
}
```
## 62. Unique Paths  
[link](https://leetcode.com/problems/unique-paths/)   
```javascript
const uniquePath = function(m, n) {
  const dp = [...Array(m)].map(_ => Array(n).fill(1))
  for (let i = 1; i < m; i++) {
    for (let j = 1; j < n; j++) {
      dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    }
  }
  return dp[m - 1][n - 1]
}
```
## 63. Unique Paths II  
[link](https://leetcode.com/problems/unique-paths-ii/)  
```javascript
const uniquePathsWithObstacles = function(obstacleGrid) {
  const m = obstacleGrid.length, n = obstacleGrid[0].length
  const dp = [...Array(m)].map(_ => Array(n).fill(1))
  dp[0][0] = obstacleGrid[0][0] == 1 ? 0 : 1
  if (dp[0][0] == 0) return 0
  for (let i = 1; i < m; i++) {
    dp[i][0] = obstacleGrid[i][0] == 1 ? 0 : dp[i - 1][0]
  }
  for (let i = 1; i < n; i++) {
    dp[0][i] = obstacleGrid[0][i] == 1 ? 0 : dp[0][i - 1]
  }
  for (let i = 1; i < m; i++) {
    for (let j = 1; j < n; j++) {
      dp[i][j] = obstacleGrid[i][j] == 1 ? 0 : dp[i - 1][j] + dp[i][j - 1]
    }
  }
  return dp[m - 1][n - 1]
}
```
## 67. Add Binary  
[link](https://leetcode.com/problems/add-binary/)
```javascript
const addBinary = function(a, b) {
  let res = ''
  let i = a.length - 1
  let j = b.length - 1
  let carry = 0
  while (i >= 0 || j >= 0 || carry > 0) {
    carry += i >= 0 ? parseInt(a[i--]) : 0
    carry += j >= 0 ? parseInt(b[j--]) : 0
    res = carry % 2 + res
    carry = parseInt(carry / 2)
  }
  return res
}
```
## 70. Climbing Stairs  
[link](https://leetcode.com/problems/climbing-stairs/)
```javascript
const climbStairs = function(n) {
  if (n < 2) return n
  let dp = new Array(n + 1).fill(0)
  dp[0] = 1
  dp[1] = 1
  for (let i = 2; i <= n; i++) {
    dp[i] = dp[i - 1] + dp[i - 2]
  }
  return dp[n]
}
```
## 83. Remove Duplicates from Sorted List  
[link](https://leetcode.com/problems/remove-duplicates-from-sorted-list/)
```javascript
const deleteDuplicates = function(head) {
  if (!head) return head
  let cur = head
  while (cur && cur.next) {
    if (cur.val === cur.next.val) cur.next = cur.next.next
    else cur = cur.next
  }
  return head
  
  // if not sorted
  let map = new Map()
  let cur = head
  let prev
  while (cur) {
    if (!map[cur.val]) {
      map[cur.val] = 1
      prev = cur
    } else prev.next = cur.next
    cur = cur.next
  }
  return head
}
```

## 88 Merge Sorted Array  
[link](https://leetcode.com/problems/merge-sorted-array/)
```javascript
const merge = function(nums1, m, nums2, n) {
  let len = m + n
  m--
  n--
  while (len--) {
    if (n < 0 || nums1[m] > nums2[n]) nums1[len] = nums1[m--]
    else nums1[len] = nums2[n--]
  }
}
```
## 92. Reverse Linked List II  
[link](https://leetcode.com/problems/reverse-linked-list-ii/)  
```javascript
const reverseBetween = function(head, m, n) {
  if (!head) return null
  let dummy = new ListNode(0)
  dummy.next = head
  let pre = dummy
  for (let i = 0; i < m - 1; i++) { pre = pre.next }
  let start = pre.next
  let then = start.next
  // insert then into pre and pre.next
  for (let i = 0; i < n - m; i++) {
    start.next = then.next
    then.next = pre.next
    pre.next = then
    then = start.next
  }
  return dummy.next
}
```
## 102. Binary Tree Level Order Traversal  
[link](https://leetcode.com/problems/binary-tree-level-order-traversal/)
```javascript
const levelOrder = function(root) {
  // dfs
  let res = []
  dfs(root, 0)
  return res

  function dfs(node, l) {
    if (!node) return
    if (!res[l]) res[l] = []
    res[l].push(node.val)
    if (node.left) dfs(node.left, l + 1)
    if (node.right) dfs(node.right, l + 1)
  }

  // bfs
  if (!root) return []
  let q = [root], res = []
  while (q.length) {
    const len = q.length
    let tmp = []
    for (let i = 0; i < len; i++) {
      let cur = q.shift()
      tmp.push(cur.val)
      if (cur.left) q.push(cur.left)
      if (cur.right) q.push(cur.right)
    }
    res.push(tmp)
  }
  return res
}
```
## 105. Construct Binary Tree from Preorder and Inorder Traversal  
[link](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
```javascript
const buildTree = function(preorder, inorder) {
  return build(0, preorder.length - 1)

  function build(l, r) {
    if (l > r) return null
    let p = preorder.shift()
    let i = inorder.indexOf(p)
    let root = new TreeNode(p)

    root.left = build(l, i - 1)
    root.right = build(i + 1, r)
    return root
  }
}
// O(N)
```
## 106. Construct Binary Tree from Inorder and Postorder Traversal  
[link](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)
```javascript
const buildTree = function(inorder, postorder) {
  return build(0, inorder.length - 1)

  function build(l, r) {
    if (l > r) return null
    let p = postorder.pop()  // ATTENTION
    let i = inorder.indexOf(p)
    let root = new TreeNode(p)

    root.right = build(i + 1, r) // ATTENTION
    root.left = build(l, i - 1)
    return root
  }
}
//O(n)
```
## 108. Convert Sorted Array to Binary Search Tree  
[link](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)
```javascript
const sortedArrayToBST = function(nums) {
  if (!nums.length) return null
  return dfs(nums, 0, nums.length - 1)

  function dfs(nums, lo, hi) {
    if (lo > hi) return null
    let mid = lo + ((hi - lo) >> 1)
    let node = new TreeNode(nums[mid])
    node.left = dfs(nums, lo, mid - 1)
    node.right = dfs(nums, mid + 1, hi)
    return node
  }
}
```
## 110. Balanced Binary Tree  
[link](https://leetcode.com/problems/balanced-binary-tree/)  
```javascript
const isBalanced = function(root) {
  return dfs(root) !== -1

  function dfs(node) {
    if (!node) return 0
    let l = dfs(node.left)
    if (l === -1) return -1
    let r = dfs(node.right)
    if (r === -1) return -1

    if (Math.abs(l - r) > 1) return -1
    return Math.min(l, r) + 1
  }
}
```
## 111. Minimum Depth of Binary Tree  
[link](https://leetcode.com/problems/minimum-depth-of-binary-tree/)  
```javascript
const minDepth = function(root) {
  // bfs
  if (!root) return 0
  let q = [root]
  let dpt = 0
  while (q.length) {
    dpt++
    let len = q.length
    for (let i = 0; i < len; i++) {
      let cur = q.shift()
      if (!cur.left && !cur.right) return dpt
      if (cur.left) q.push(cur.left)
      if (cur.right) q.push(cur.right)
    }
  }
  // dfs
  if (!root) return 0
  let l = minDepth(root.left)
  let r = minDepth(root.right)
  return 1 + Math.min(l, r) || Math.max(l, r)
}
```
## 112. Path Sum  
[link](https://leetcode.com/problems/path-sum/)  
```javascript
const hasPathSum = function(root, sum) {
  if (!root) return false
  if (!root.left && !root.right) return sum === root.val
  return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val)
}
```
## 113. Path Sum II  
[link](https://leetcode.com/problems/path-sum-ii/)
```javascript
const pathSum = function(root, sum) {
  let res = []
  dfs(root, 0, [])
  return res

  function dfs(node, cur, arr) {
    if (!node) return
    arr.push(node.val)
    cur += node.val
    if (!node.left && !node.right) {
      if (cur === sum) res.push(arr.slice())
      return
    }
    dfs(node.left, cur, arr.slice())
    dfs(node.right, cur,arr.slice())
  }
}
```
## 121. Best Time to Buy and Sell Stock  
[link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)  
```javascript
const maxProfit = function(prices) {
  let max = 0, low = prices[0]
  prices.map(p => {
    max = Math.max(max, p - low)
    low = Math.min(low, p)
  })
  return max
}
```
## 122. Best Time to Buy and Sell Stock II  
[link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)  
```javascript
var maxProfit = function(prices) {
  let max = 0
  for (let i = 1; i < prices.length; i++) {
    let prev = prices[i - 1]
    let cur = prices[i]
    if (prev < cur) max += cur - prev
  }
  return max
}
```  
## 123. Best Time to Buy and Sell Stock III  
[link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/)  
```javascript
const maxProfit = function(prices) {
  let hold1 = -Infinity, hold2 = -Infinity
  let sell1 = 0, sell2 = 0
  prices.map(p => {
    hold1 = Math.max(hold1, -p)
    sell1 = Math.max(sell1, hold1 + p)
    hold2 = Math.max(hold2, sell1 - p)
    sell2 = Math.max(sell2, hold2 + p)
  })
  return sell2
};
```
## 124. Binary Tree Maximum Path Sum  
[link](https://leetcode.com/problems/binary-tree-maximum-path-sum/)  
```javascript
const maxPathSum = function(root) {
  // dfs  
  let max = -Infinity
  dfs(root)
  return max

  function dfs(node) {
    if (!node) return 0
    let l = Math.max(0, dfs(node.left))
    let r = Math.max(0, dfs(node.right))
    max = Math.max(max, l + r + node.val)
    return Math.max(l, r) + node.val
  }
}
```
## 128. Longest Consecutive Sequence  
[link](https://leetcode.com/problems/longest-consecutive-sequence/)  
```javascript
const longestConsecutive = function(nums) {
  let set = new Set(nums)
  let cur = 0, max = 0
  for (let num of nums) {
    if (!set.has(num - 1)) {
      let tmp = num
      while (set.has(tmp++)) {
        cur++
        max = Math.max(max, cur)
      }
      cur = 0
    }
  }
  return max
}
```
## 139. Word Break  
[link](https://leetcode.com/problems/word-break/)  
```javascript
const wordBreak = function(s, wordDict) {
  let len = s.length, dp = new Array(len + 1)
  for (let i = 0; i <= len; i++) { dp[i] = false }
  dp[0] = true
  for (let i = 1; i <= len; i++) {
    for (let j = 0; j < i; j++) {
      if (dp[j] && wordDict.indexOf(s.substring(j, i)) >= 0) {
        dp[i] = true
        break
      }
    }
  }
```
## 141. Linked List Cycle  
[link](https://leetcode.com/problems/linked-list-cycle/)
```javascript
const hasCycle = function(head) {
  let walker = head
  let runner = head
  while (runner && runner.next) {
    walker = walker.next
    runner = runner.next.next
    if (walker == runner) return true
  }
  return false
}

```
## 143. Reorder List  
[link](https://leetcode.com/problems/reorder-list/)  
```javascript
const reorderList = function(head) {
  if (!head || !head.next) return
  
  let fast = head
  let slow = head
  while (fast.next && fast.next.next) {
    slow = slow.next
    fast = fast.next.next
  }

  // split
  let part2 = slow.next
  slow.next = null

  // reverse part 2
  let prev = null, cur = part2, next = null
  while (cur) {
    next = cur.next
    cur.next = prev
    prev = cur
    cur = next
  }
  part2 = prev

  // merge
  while (head && part2) {
    let p1 = head.next
    let p2 = part2.next
    head.next = part2
    head.next.next = p1
    part2 = p2
    head = p1
  }
}
```
## 146. LRU Cache  
[link](https://leetcode.com/problems/lru-cache/)  
```javascript
var Node = function(key, value) {
  this.key = key
  this.val = value
  this.prev = this.next = null
}

var DoublyLinkedList = function() {
  this.head = new Node
  this.tail = new Node
  this.head.next = this.tail
  this.tail.prev = this.head
}
// insert a node right after head
DoublyLinkedList.prototype.insertHead = function(node) {
  node.prev = this.head
  node.next = this.head.next
  this.head.next.prev = node
  this.head.next = node
}
// remove node from linked list
DoublyLinkedList.prototype.removeNode = function(node) {
  let prev = node.prev
  let next = node.next
  prev.next = next
  next.prev = prev
}
// move a node to the head
DoublyLinkedList.prototype.moveToHead = function(node) {
  this.removeNode(node)
  this.insertHead(node)
}
// remove the tail element and return its key
DoublyLinkedList.prototype.removeTail = function() {
  let tail = this.tail.prev
  this.removeNode(tail)
  return tail.key
}
var LRUCache = function(capacity) {
  this.capacity = capacity
  this.currentSize = 0
  this.hash = new Map()
  this.dll = new DoublyLinkedList()
};

LRUCache.prototype.get = function(key) {
  let node = this.hash.get(key)
  if (!node) return -1
  this.dll.moveToHead(node)
  return node.val
};

LRUCache.prototype.put = function(key, value) {
  let node = this.hash.get(key)
  if (!node) {
    let newNode = new Node(key, value)
    this.hash.set(key, newNode)
    this.dll.insertHead(newNode)
    this.currentSize++
    if (this.currentSize > this.capacity) {
      let tailKey = this.dll.removeTail()
      this.hash.delete(tailKey)
      this.currentSize--
    }
  } else {
    node.val = value
    this.dll.moveToHead(node)
  }
};
```
## 155. Min Stack  
[link](https://leetcode.com/problems/min-stack/)  
```javascript
const MinStack = function() {
  this.stack = []
};

MinStack.prototype.push = function(x) {
  let min = this.stack.length === 0 ? x : this.stack[this.stack.length - 1].min 
  this.stack.push({val: x, min:  Math.min(min, x)})
};

MinStack.prototype.pop = function() {
  if (this.stack.length > 0) this.stack.pop() 
};

MinStack.prototype.top = function() {
   if (this.stack.length > 0) return this.stack[this.stack.length - 1].val 
};

MinStack.prototype.getMin = function() {
  if (this.stack.length > 0) return this.stack[this.stack.length - 1].min 
};
```
## 160. Intersection of Two Linked Lists  
[link](https://leetcode.com/problems/intersection-of-two-linked-lists/)  
```javascript
const getIntersectionNode = function(headA, headB) {
  if (!headA || !headB) return null
  let a = headA
  let b = headB
  while (a != b) {
    a = a == null ? headB : a.next
    b = b == null ? headA : b.next
  }
  return a
}
```
## 188. Best Time to Buy and Sell Stock IV  
[link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/)
```javascript
const maxProfit = function(k, prices) {
  if (k >= prices.length / 2 | 0)  {
    let profit = 0
    for (let i = 1; i < prices.length; i++) {
      if (prices[i] > prices[i - 1]) profit += prices[i] - prices[i - 1]
    }
    return profit
  }
  let buy = new Array(k + 1).fill(-Infinity), sell = new Array(k + 1).fill(0)
  prices.map(p => {
    for (let i = 1; i <= k; i++) {
      buy[i] = Math.max(buy[i], sell[i - 1] - p)
      sell[i] = Math.max(sell[i], buy[i] + p)
    }
  })
  return sell[k]
}
```
## 199. Binary Tree Right Side View  
[link](https://leetcode.com/problems/binary-tree-right-side-view/)
```javascript
const rightSideView = function(root) {
  // dfs
  const res = []
  dfs(root, 0)
  return res

  function dfs(node, l) {
    if (!node) return
    res[l] = node.val
    dfs(node.left, l + 1)
    dfs(node.right, l + 1)
  }

  // bfs
  if (!root) return []
  const res = [root.val], q = [root]
  while (q.length) {
    let nxt = []
    while (q.length) {
      let node = q.shift()
      if (node.left) nxt.push(node.left)
      if (node.right) nxt.push(node.right)
    }
    if (nxt[nxt.length - 1]) res.push(nxt[nxt.length - 1].val)
    q = nxt
  }
  return res

}
```
## 200. Number of Islands  
[link](https://leetcode.com/problems/number-of-islands/)  
```javascript
const numIslands = function(grid) {
  if (!grid || !grid.length || !grid[0].length) return 0
  const m = grid.length, n = grid[0].length
  let cnt = 0
  for (let i = 0; i < m; i++) {
    for (let j = 0; i < n; j++) {
      if (grid[i][j] == 1) {
        dfs(i, j)
        cnt++
      }
    }
  }
  return cnt

  function dfs(x, y) {
    if (x < 0 || y < 0 || x >= m || y >= n || grid[x][y] !== '1') return 
    grid[x][y] = '0'
    let directions = [-1, 0, 1, 0, -1]
    for (let i = 0; i < directions.length - 1; i++) {
      let newX = x + directions[i]
      let newY = y + directions[i + 1]
      dfs(newX, newY)
    }
  }
}
```
## 206. Reverse Linked List  
[link](https://leetcode.com/problems/reverse-linked-list/)
```javascript
var reverseList = function(head) {
  // iterative
  let prev = null
  while (head) {
    let next = head.next
    head.next = prev
    prev = head
    head = next
  }
  return prev
  
  // recursive
  if (!head || !head.next) return head
  let newHead = reverseList(head.next)
  head.next.next = head
  head.next = null
  return newHead
}
```
## 215. Kth Largest Element in an Array  // TODO
[link](https://leetcode.com/problems/kth-largest-element-in-an-array/)  
```javascript
const findKthLargest = function(nums, k) {
  k = nums.length - k
  let lo = 0, hi = nums.length - 1
  while (lo <= hi) {
    let j = partition(nums, lo, hi)
    if (j < k) lo = j + 1
    else if (j > k) hi = j - 1
    else break
  }
  return nums[k]
  
  function partition(nums, lo, hi) {
    let i = lo, j = hi + 1
    while (true) {
      while (i < hi && nums[++i] < nums[lo]) {}
      while (j > lo && nums[lo] < nums[--j]) {}
      if (i >= j) break
      swap(i, j)
    }
    swap(lo, j)
    return j
  }

  function swap(i, j) {
    let tmp = nums[i]
    nums[i] = nums[j]
    nums[j] = tmp
  }
}
// O(N)  worst O(N^2)
```
## 230. Kth Smallest Element in a BST  
[link](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)
```javascript
const kthSmallest = function(root, k) {
  let cnt = 0, res = 0
  traverse(root, k)
  return res

  function traverse(node, k) {
    if (!node) return
    traverse(node.left, k)
    cnt++
    if (k === cnt) return res = node.val
    traverse(node.right, k)
  }
}
```
## 234. Palindrome Linked List  
[link](https://leetcode.com/problems/palindrome-linked-list/)  
```javascript
const isPalindrome = function(head) {
  let fast = head
  let slow = head
  while (fast && fast.next) {
    fast = fast.next.next
    slow = slow.next
  }
  let half = reverse(slow)
  while (head && half) {
    if (head.val !== half.val) return false
    head = head.next
    half = half.next
  }
  return true
  
  function reverse(head) {
    let prev = null
    while (head) {
      let next = head.next
      head.next = prev
      prev = head
      head = next
    }
    return prev
  }
}
```
## 236. Lowest Common Ancestor of a Binary Tree  
[link](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)  
```javascript
const lowestCommonAncestor = function(root, p, q) {
  if (!root || root == p || root == q) return root
  let left = lowestCommonAncestor(root.left, p, q)
  let right = lowestCommonAncestor(root.right, p, q)
  return (left && right) ? root : (left || right)
}
```
## 297. Serialize and Deserialize Binary Tree  
[link](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)
```javascript
const serialize = function(root) {
  let data = []
  return dfs(root, data)
  
  function dfs(node, data) {
    if (!node) data.push(null)
    else {
      data.push(node.val)
      dfs(node.left, data)
      dfs(node.right, data)
    }
    return data
  }
}

const deserialize = function(data) {
  return helper()

  function helper() {
    if (data.length === 0) return
    let next = data.shift()
    if (next == null) return null
    else {
      let node = new TreeNode(next)
      node.left = helper()
      node.right = helper()
      return node
    }
  }
}
```
## 309. Best Time to Buy and Sell Stock with Cooldown  
[link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)  
```javascript
const maxProfit = function(prices) {
  let prevBuy = 0, buy = -prices[0], prevSell = 0, sell = 0
  prices.map(price => {
    prevBuy = buy
    buy = Math.max(prevBuy, prevSell - price)
    prevSell = sell
    sell = Math.max(prevSell, prevBuy + price)
  })
  return sell
}
```
## 322. Coin Change  
[link](https://leetcode.com/problems/coin-change/)  
```javascript
const coinChange = function(coin, amount) {
  let dp = new Array(amount + 1).fill(amount + 1)
  dp[0] = 0
  for (let coin of coins) {
    for (let i = coin; i <= amount; i++) {
      dp[i] = Math.min(dp[i], dp[i - coin] + 1)
    }
  }
  return dp[amount] > amount ? -1 : dp[amount]
}
```
## 344. Reverse String  
[link](https://leetcode.com/problems/reverse-string/)  
```javascript
const reverseString = function(s) {
  let i = 0, j = s.length - 1
  while (i < j) {
    let tmp = s[i]
    s[i] = s[j]
    s[j] = tmp
    i++
    j--
  }
}
```
## 347. Top K Frequent Elements // TODO
[link](https://leetcode.com/problems/top-k-frequent-elements/)  
```javascript
const topKFrequent = function(nums, k) {
  const map = new Map()
  const res = []
  const bucket = [...Array(nums.length + 1)].map(_ => [])
  for (let num of nums) {
    map[num] = ~~map[num] + 1
  }
  for (let num in map) {
    bucket[map[num]].push(num | 0)
  }
  for (let i = nums.length; i >= 0 && k > 0; k--) {
    while (bucket[i].length === 0) i--
    res.push(bucket[i].shift())
  }
  return res
}
```
## 415. Add Strings  
[link](https://leetcode.com/problems/add-strings/)  
```javascript
const addStrings = function(num1, num2) {
  let m = num1.length - 1
  let n = num2.length - 1
  let carry = 0
  let res = ''
  while (m >= 0 || n >= 0 || carry) {
    carry += m >= 0 ? num1[m--] | 0 : 0
    carry += n >= 0 ? num2[n--] | 0 : 0
    res = carry % 10 + res
    carry = carry / 10 | 0
  }
  return res
}
```
## 445. Add Two Numbers II  
[link](https://leetcode.com/problems/add-two-numbers-ii/)  
```javascript
const addTwoNumbers = function(l1, l2) {
  let s1 = [], s2 = []
  while (l1) {
    s1.push(l1.lal)
    l1 = l1.next
  }
  while (l2) {
    s2.push(l2.lal)
    l2 = l2.next
  }

  let dummy = new ListNode(0)
  let cur = null, next = null
  let carry = 0
  while (s1.length || s2.length || carry) {
    carry += s1.length ? s1.pop() : 0
    carry += s2.length ? s2.pop() : 0
    cur = new ListNode(carry % 10)
    carry = carry / 10 | 0
     // just insert the new cur between dummy and next
    dummy.next = cur
    cur.next = next
    next = cur
  }
  return dummy.next

}
```
## 496. Next Greater Element I  
[link](https://leetcode.com/problems/next-greater-element-i/)  
```javascript
const nextGreaterElemenet = function(nums1, nums2) {
  let map = new Map()
  let stack = []
  for (let num of nums2) {
    while (stack.length && stack[stack.length - 1] < num) {
      map.set(stack.pop(), num)
    }
    stack.push(num)
  }
  for (let i = 0; i < nums1.length; i++) {
    nums1[i] = map.has(nums1[i]) ? map.get(nums1[i]) : -1
  }
  return nums1
}
// O(N)
```
## 503. Next Greater Element II  
[link](https://leetcode.com/problems/next-greater-element-ii/)  
```javascript
const nextGreaterElements = function(nums) {
  const n = nums.length, next = new Array(n).fill(-1)
  const stack = []
  for (let i = 0; i < n * 2; i++) {
    let num = nums[i % n]
    while (stack.length && nums[stack[stack.length - 1]] < num) { 
      next[stack.pop()] = num
    }
    if (i < n) stack.push(i)
  }
  return next
}

```
## 509. Fibonacci Number  
[link](https://leetcode.com/problems/fibonacci-number/)  
```javascript
const fib = function(N) {
  const dp = new Array(N + 1).fill(0)
  dp[1] = 1
  for (let i = 2; i <= N; i++) {
    dp[i] = dp[i - 1] + dp[i - 2]
  }
  return dp[N]
}
```
## 515. Find Largest Value in Each Tree Row  
[link](https://leetcode.com/problems/find-largest-value-in-each-tree-row/)
```javascript
const largestValues = function(root) {
  let res = []
  dfs(root, 0)
  return res
  
  function dfs(node, l) {
    if (!node) return
    if (!res[l] && res[l] !== 0) res[l] = node.val
    else res[l] = Math.max(res[l], node.val)
    dfs(node.left, l + 1)
    dfs(node.right, l + 1)
  }
}
```
## 518. Coin Change II  
[link](https://leetcode.com/problems/coin-change-2/)
```javascript
const change = function(amount, coins) {
  let dp = new Array(amount + 1).fill(0)
  dp[0] = 1
  for (let c of coins) {
    for (let i = c; i <= amount; i++) dp[i] += dp[i - c]
  }
  return dp[amount]
}
```
## 543. Diameter of Binary Tree  
[link](https://leetcode.com/problems/diameter-of-binary-tree/)
```javascript
const diameterOfBinaryTree = function(root) {
  return dpt(root.left) + dpt(root.right)
  
  function dpt(node) {
    if (!node) return 0
    return 1 + Math.max(dpt(node.left), dpt(node.right))
  }
}
```
## 695.  Max Area of Island  
[link](https://leetcode.com/problems/max-area-of-island/)  
```javascript
const maxAreaOfIsland = function(grid) {
  if (!grid || !grid.length || !grid[0].length) return 0
  const m = grid.length, n = grid[0].length
  let max = 0
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      if (grid[i][j] == 1) max = Math.max(dfs(i, j), max)
    }
  }
  return max

  function dfs(x, y) {
    grid[x][y] = 0
    let sum = 1
    const directions = [-1, 0, 1, 0, -1]
    for (let i = 0; i < directions.length - 1; i++) {
      let newX = x + directions[i]
      let newY = y + directions[i + 1]
      if (!outOfBound(newX, newY) && grid[newX][newY] == 1) sum += dfs(newX, newY)
    }
    return sum
  }

  function outOfBound(x, y) {
    return x < 0 || y < 0 || x >= m || y >= n
  }

}
```

## 714. Best Time to Buy and Sell Stock with Transaction Fee  
[link](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)
```javascript
const maxProfit = function(prices, fee) {
  let buy = -prices[0], preBuy = 0, preSell = 0, sell = 0
  prices.map(p => {
    preBuy = buy
    buy = Math.max(preSell - p, preBuy)
    sell = Math.max(preBuy + p - fee, preSell)
    preSell = sell
  })
  return sell
}
```
## 814. Binary Tree Pruning  
[link](https://leetcode.com/problems/binary-tree-pruning/)  
```javascript
const pruneTree = function(root) {
  if (!root) return null
  const left = pruneTree(root.left)
  const right = pruneTree(root.right)
  if (!left) root.left = left
  if (!right) root.right = right

  return (!root.left && !root.right && root.val === 0) ? null : root
}
```
## 958. Check Completeness of a Binary Tree  
[link](https://leetcode.com/problems/check-completeness-of-a-binary-tree/)  
```javascript
const isCompleteTree = function(root) {
  if (!root) return true
  let end = false
  let q = [root]
  while (q.length) {
    let cur = q.shift()
    if (cur == null) end = true
    else {
      if (end) return false
      q.push(cur.left)
      q.push(cur.right)
    }
  }
  return true
}
```
## 994. Rotting Oranges  
[link](https://leetcode.com/problems/rotting-oranges/)  
```javascript
const orangesRotting = function(grid) {
  let q = []
  let mins = 0
  let fresh = 0
  for (let i = 0; i < grid.length; i++) {
    for (let j = 0; j < grid[0].length; j++) {
      if (grid[i][j] == 1) fresh++
      if (grid[i][j] == 2) q.push([i, j])
    }
  }
  
  while (q.length && fresh) {
    const directions = [-1, 0, 1, 0, -1]
    
    let next = []
    while (q.length) {
      let [x, y] = q.shift()
      for (let i = 0; i < directions.length; i++) {
        let newX = x + directions[i]
        let newY = y + directions[i + 1]
        if (!outOfBound(newX, newY)) {
          if (grid[newX][newY] == 1) {
            grid[newX][newY] = 2
            fresh--
            next.push([newX, newY])
          }
        }
      }
    }
    mins++
    q = next
  }
  return fresh == 0 ? mins : -1
  
  function outOfBound(x, y) {
    return x < 0 || y < 0 || x >= grid.length || y >= grid[0].length
  }
}
```
## 977. Squares Of a Sorted Array  
[link](https://leetcode.com/problems/squares-of-a-sorted-array/)  
```javascript
const sortedSquares = function(A) {
  const n = A.length
  let res = []
  let i = 0, j = n - 1
  for (let k = n - 1; k >= 0; k--) {
    if (Math.abs(A[i]) > Math.abs(A[j])) {
      res[k] = A[i] * A[i]
      i++
    } else {
      res[k] = A[j] * A[j]
      j--
    }
  }
  return res
}
```
## 1143. Longest Common Subsequence  
[link](https://leetcode.com/problems/longest-common-subsequence/)
```javascript
const longestCommonSubsequence = function(text1, text2) {
  let dp = [...new Array(text1.length + 1)].map(_ => new Array(text2.length + 1).fill(0))
  for (let i = 1; i < dp.length; i++) {
    for (let j = 1; j < dp[0].length; j++) {
      if (text1[i - 1] == text2[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1
      else dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    }
  }
  return dp[dp.length - 1][dp[0].length - 1]
}
```
## 1299. Replace Elements with Greatest Element on Right Side  
[link](https://leetcode.com/problems/replace-elements-with-greatest-element-on-right-side/)
```javascript
const replaceElements = function(arr) {
  const res = new Array(arr.length)
  res[arr.length - 1] = -1

  for (let i = arr.length - 1; i > 0; i--) {
    res[i - 1] = Math.max(arr[i], res[i])
  }
  return res
}
```

