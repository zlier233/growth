# Hot Algo  


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
    let p = postorder.pop()
    let i = inorder.indexOf(p)
    let root = new TreeNode(p)

    root.right = build(i + 1, r)
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
  while (lo < hi) {
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
## 518. Coin Change II  
[link]()
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
## 1143. Longest Common Subsequence  
[link]()
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
