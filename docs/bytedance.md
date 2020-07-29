# Hot Algo  
## 1. Two Sum  
[link](https://leetcode.com/problems/two-sum/)  
```javascript
const twoSum = function(nums, target) {
  const map = new Map()
  for (let i = 0; i < nums.length; i++) {
    if (map.has(target - nums[i])) return [map.get(target - nums[i]), i]
    map.set(nums[i], i)
  }
}
```
## 2. Add Two Numbers  
[link](https://leetcode.com/problems/add-two-numbers/)  
```javascript
const addTwoNumbers = function(l1, l2) {
  let carry = 0
  let dummy = new ListNode(0)
  let cur = dummy
  
  while (l1 || l2 || carry) {
    carry += l1 ? l1.val : 0
    carry += l2 ? l2.val : 0
    cur.next = new ListNode(carry % 10)
    cur = cur.next
    carry = carry / 10 | 0
    if (l1) l1 = l1.next
    if (l2) l2 = l2.next
  }
  return dummy.next
}
```
## 3. Longest Substring Without Repeating Characters  
[link](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
```javascript
const lengthOfLongestSubstring = function(s) {
  let map = new Map()
  let head = -1
  let max = 0
  
  for (let i = 0; i < s.length; i++) {
    if (map.has(s[i]) && map.get(s[i]) >= head) head = map.get(s[i])
    map.set(s[i], i)
    max = Math.max(i - head, max)
  }
  return max
}
```
## 4. Median Of Two Sorted Arrays  
[link](https://leetcode.com/problems/median-of-two-sorted-arrays/)  
```javascript
const findMedian = function(nums1, nums2) {
  let m = nums1.length, n = nums2.length
  if (m > n) {
    let tmp = m, tmpNum = nums1
    m = n
    nums1 = nums2
    n = tmp
    nums2 = tmpNum
  }
  let maxLeft = 0, minRight = 0
  let lo = 0, hi = m, halfLen = (m + n + 1) / 2 | 0
  while (lo <= hi) {
    let i = lo + ((hi - lo) >> 1), j = halfLen - i
    if (i > 0 && nums1[i - 1] > nums2[j]) hi = i - 1
    else if (j > 0 && nums2[j - 1] > nums1[i]) lo = i + 1
    else {
      if (i === 0) maxLeft = nums2[j - 1]
      else if (j === 0) maxLeft = nums1[i - 1]
      else maxLeft = Math.max(nums1[i - 1], nums2[j - 1])
      if ((m + n) & 1) return maxLeft
      
      if (i === m) minRight = nums2[j]
      else if (j === n) minRight = nums1[i]
      else minRight = Math.min(nums1[i], nums2[j])
      return (maxLeft + minRight) / 2.0
    }
  }
}
```
## 8. String to Integer (atoi)  
[link](https://leetcode.com/problems/string-to-integer-atoi/)  
```javascript
const myAtoi = function(str) {
  let filter = '0123456789+- '
  let sign = 1
  let res = 0
  for (let char of str) {
    let idx = filter.indexOf(char) 
    if (idx >= 0) {
      if (char === ' ') continue
      if (filter[10] === '+') filter = filter.slice(0, 10)
      if (char === '+') continue
      if (char === '-') { sign = -sign; continue; }
      res = 10 * res + idx
    } else break
  }
  res = res * sign
  if (res > Math.pow(2, 31) - 1) return Math.pow(2, 31) - 1
  if (res < -Math.pow(2, 31)) return -Math.pow(2, 31)
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
    if (nums[i] > 0) return arr
    if (i > 0 && nums[i] === nums[i - 1]) continue
    for (let j = i + 1, k = nums.length - 1; j < k;) {
      if (nums[i] + nums[j] + nums[k] === 0) {
        arr.push([nums[i], nums[j], nums[k]])
        j++
        k--
        while (j < k && nums[j] === nums[j - 1]) { j++ }
        while (j < k && nums[k] === nums[k + 1]) { k-- }
      } else if (nums[i] + nums[j] + nums[k] > 0) k--
      else j++
    }
  }
  return arr
}
```
## 18. 4Sum  
[link](https://leetcode.com/problems/4sum/)  
```javascript
  let arr = []
  nums = nums.sort((a, b) => a - b)

  for (let i = 0; i < nums.length - 3; i++) {
    if (i > 0 && nums[i] == nums[i - 1]) continue
    for (let j = i + 1; j < nums.length - 2; j++) {
      if (j > i + 1 && nums[j] == nums[j - 1]) continue
      for (let m = j + 1, n = nums.length - 1; m < n;) {
        if (nums[i] + nums[j] + nums[m] + nums[n] == target) {
          arr.push([nums[i], nums[j], nums[m], nums[n]])
          m++
          n--
          while (m < n && nums[m - 1] == nums[m]) m++
          while (m < n && nums[n] == nums[n + 1]) n--
        }
        else if (nums[i] + nums[j] + nums[m] + nums[n] - target > 0) n--
        else m++
      }
    }
  }
  return arr
}
```
## 20. Valid Parentheses  
[link](https://leetcode.com/problems/valid-parentheses/)  
```javascript
const isValid = function(s) {
  let map = {
    '(': ')',
    '{': '}',
    '[': ']'
  }
  const stack = []
  for (let i = 0; i < s.length; i++) {
    if (map[s[i]]) stack.push(map[s[i]])
    else {
      if (s[i] !== stack.pop()) return false
    }
  }
  return stack.length === 0
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
    let cur = dummy
    while (a && b) {
      if (a.val < b.val) {
        cur.next = a
        a = a.next
      } else {
        cur.next = b
        b = b.next
      }
      cur = cur.next
    }
    if (a) cur.next = a
    if (b) cur.next = b
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
  if (!head) return head
  let dummy = new ListNode(0)
  dummy.next = head
  let pointer = dummy
  while (pointer) {
    let node = pointer
    for (let i = 0; i < k && node; i++) {
      node = node.next
    }
    if (!node) break
    let pre = null, cur = pointer.next, next = null
    for (let i = 0; i < k; i++) {
      next = cur.next
      cur.next = pre
      pre = cur
      cur = next
    }
    let tail = pointer.next
    tail.next = cur
    pointer.next = pre
    pointer = tail
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
const firstMissingPositive = function(nums) {
  // put each num in right place
  // missing num in range 1 ~ nums.length
  for (let i = 0; i < nums.length; i++) {
    while (nums[i] > 0 && nums[i] <= nums.length && nums[i] != nums[nums[i] - 1]) { // ATT
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
## 48. Rotate Image  
[link](https://leetcode.com/problems/rotate-image/)  
```javascript
const rotate = function(matrix) {
  // clockwise
  // first up and down reverse
  // then anti-diag reverse
  let m = matrix.length
  let n = matrix[0].length
  let midM = matrix.length >> 1
  for (let i = 0; i < midM; i++) {
    for (let j = 0; j < n; j++) {
      swap(i, j, m - i - 1, j)
    }
  }
  
  for (let i = 0; i < m; i++) {
    for (let j = i + 1; j < n; j++) {
      swap(i, j, j, i)
    }
  }
  
  
  function swap(x, y, m, n) {
    let tmp = matrix[x][y]
    matrix[x][y] = matrix[m][n]
    matrix[m][n] = tmp
  }
  // anti-clockwise
  // first left and right reverse
  // then anti-diag reverse
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
  dp[0][0] = obstacleGrid[0][0] == 1 ? 0 : 1   // ATT
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
## 69. Sqrt(x)  
[link](https://leetcode.com/problems/sqrtx/)  
```javascript
const mySqrt = function(x) {
  // math
  let hi = x
  while (hi * hi > x) {
    hi = (hi + x / hi) >> 1
  }
  return hi
  // binary search
  let lo = 0, hi = x
  while (lo < hi) {
    let mid = lo + ((hi - lo) >> 1)
    if (mid * mid === x) return mid
    else if (x > mid * mid) lo = mid + 1
    else hi = mid
  }
  return x < lo * lo ? lo - 1 : lo
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
  // memo
  if (n <= 0) return 0
  if (n === 1) return 1
  if (n === 2) return 2
  let oneStep = 2
  let twoStep = 1
  let sum = 0
  for (let i = 2; i < n; i++) {
    sum = oneStep + twoStep
    twoStep = oneStep
    oneStep = sum
  }
  return sum
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

## 88. Merge Sorted Array  
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
## 100. Same Tree  
[link](https://leetcode.com/problems/same-tree/)  
```javascript
const isSame = function(p, q) {
  // bfs
  let stack = [[p, q]]
  while (stack.length) {
    let [n1, n2] = stack.pop()
    if (!n1 && !n2) continue
    if (!n1 || !n2 || n1.val !== n2.val) return false
    stack.push([n1.left, n2.left])
    stack.push([n1.right, n2.right])
  }
  return true
  
  // dfs
  if (!p || !q) return p == q
  return p.val === q.val && isSame(p.left, q.left) && isSame(p.right, q.right)

}
```
## 101. Symmetric Tree  
[link](https://leetcode.com/problems/symmetric-tree/)  
```javascript
const isSymmetric = function(root) {
  // bfs
  if (!root) return true
  let stack = [[root.left, root.right]]
  while (stack.length) {
    let [l, r] = stack.shift()
    if (!l && !r) continue
    if (!l || !r || l.val !== r.val) return false
    stack.push([l.left, r.right])
    stack.push([l.right, r.left])
  }
  return true
  // dfs
  if (!root) return true
  return isMirror(root.left, root.right)

  function isMirror(l, r) {
    if (!l || !r) return l == r
    return l.val === r.val && isMirror(l.left, r.right) && isMirror(l.right, r.left)
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
    let tmp = []
    let next = []
    while (q.length) {
      let cur = q.shift()
      tmp.push(cur.val)
      if (cur.left) next.push(cur.left)
      if (cur.right) next.push(cur.right)
    }
    res.push(tmp)
    q = next

  }
  return res
}
```
## 103. Binary Tree Zigzag Level Order Traversal  
[link](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)  
```javascript
const zigzagLevelOrder = function(root) {
  // dfs
  if (!root) return []
  const res = []
  dfs(root, 0)
  return res

  function dfs(node, l) {
    if (!node) return
    if (!res[l]) res[l] = []
    if (l & 1) res[l].unshift(node.val)
    else res[l].push(node.val)
    dfs(node.left, l + 1)
    dfs(node.right, l + 1)
  }

  // bfs
  if (!root) return []
  let q = [root]
  const res = []
  let zigzag = true

  while (q.length) {
    const len = q.length
    const tmp = [], next = []
    while (q.length) {
      let cur = q.pop() // ATT
      tmp.push(cur.val)
      if (zigzag) {  // ATT
        if (cur.left) next.push(cur.left)
        if (cur.right) next.push(cur.right)
      } else {
        if (cur.right) next.push(cur.right)
        if (cur.left) next.push(cur.left)
      }
    }
    res.push(tmp)
    zigzag = !zigzag
    q = next
  }
  return res
}
```
## 104. Maximum Depth Of Binary Tree  
[link](https://leetcode.com/problems/maximum-depth-of-binary-tree/)  
```javascript
const maxDepth = function(root) {
  // bfs
  if (!root) return 0
  let q = [root]
  let cnt = 0
  while (q.length) {
    let nxt = []
    while (q.length) {
      let cur = q.shift()
      if (cur.left) nxt.push(cur.left)
      if (cur.right) nxt.push(cur.right)
    }
    q = nxt 
    cnt++
  }
  return cnt
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
    return Math.max(l, r) + 1
  }
}
```
## 111. Minimum Depth of Binary Tree  
[link](https://leetcode.com/problems/minimum-depth-of-binary-tree/)  
```javascript
const minDepth = function(root) {
  // bfs
  if (!root) return 0
  let cnt = 0
  let q = [root]
  while (q.length) {
    let nxt = []
    cnt++
    while (q.length) {
      let cur = q.shift()
      if (!cur.left && !cur.right) return cnt
      if (cur.left) nxt.push(cur.left)
      if (cur.right) nxt.push(cur.right)
    }
    q = nxt
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
## 114. Flatten Binary Tree to Linked List  
[link](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/)  
```javascript
const flatten = function(root) {
  let cur = root
  while (cur) {
    if (cur.left) {
      let pre = cur.left
      while (pre.right) {pre = pre.right}
      pre.right = cur.right
      cur.right = cur.left
      cur.left = null
    }
    cur = cur.right
  }
}
// O(N)
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
    sell2 = Math.max(sell2, hold2 + p)
    hold2 = Math.max(hold2, sell1 - p)
    sell1 = Math.max(sell1, hold1 + p)
    hold1 = Math.max(hold1, -p)
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
    if (!set.has(num - 1)) { // make sure it's the min 
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
## 136. Single Number  
[link](https://leetcode.com/problems/single-number/)  
```javascript
const singleNumber = function(nums) {
  // map
  let map = new Map()
  for (let num of nums) {
    if (!map.has(num)) {
      map.set(num, 1) 
    } else {
      map.delete(num)
    }
  }
  return map.keys().next().value
  // xor
  let res = nums[0]
  for (let i = 1; i < nums.length; i++) {
    res ^= nums[i]
  }
  return res
}
```
## 138. Copy List with Random Pointer  
[link](https://leetcode.com/problems/copy-list-with-random-pointer/)  
```javascript
const copyRandomList = function(head) {
  let pointer = head
  let next = null

  while (pointer) {
    next = pointer.next
    let copy = new Node(pointer.val)
    pointer.next = copy
    copy.next = next

    pointer = next
  }

  pointer = head
  while (pointer) {
    if (pointer.random) {
      pointer.next.random = pointer.random.next
    }
    pointer = pointer.next.next
  }

  pointer = head
  let dummy = new Node(0)
  let copy = null
  let copyPointer = dummy
  while (pointer) {
    next = pointer.next.next
    // extract copy
    copy = pointer.next
    copyPointer.next = copy
    copyPointer = copy
    // restore the origin list
    pointer.next = next
    pointer = next
  }
  return dummy.next
}
```
## 139. Word Break  
[link](https://leetcode.com/problems/word-break/)  
```javascript
const wordBreak = function(s, wordDict) {
  // dp[i] means whether s[0, i - 1] could be sperated by words in wordDict
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
  return dp[len]
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
## 142. Linked List Cycle II  
[link](https://leetcode.com/problems/linked-list-cycle-ii/)  
```javascript
const detectCycle = function(head) {
  let slow = head
  let fast = head
  while (fast && fast.next) {
    fast = fast.next.next
    slow = slow.next
    if (fast == slow) {
      let tool = head
      while (tool != slow) {
        tool = tool.next
        slow = slow.next
      }
      return tool
    }
  }
  return null
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
## Odd Increase Even Decrease  
```javascript
const reord = function(head) {
  if (!head || !head.next) return head
  let odd = head
  let even = head.next
  let curOdd = odd
  let curEven = even
  while (curOdd && curOdd.next) {
    curOdd.next = curOdd.next.next
    curOdd = curOdd.next
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

## 162. Find Peak Element  
[link](https://leetcode.com/problems/find-peak-element/)  
```javascript
const findPeakElement = function(nums) {
  let lo = 0, hi = nums.length - 1
  while (lo < hi) {
    let mid = lo + ((hi - lo) >> 1)
    // make each bound true
    if (nums[mid] < nums[mid + 1]) lo = mid + 1
    else hi = mid
  }
  return lo
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
  let q = [root]
  let res = [root.val]
  while (q.length) {
    const len = q.length
    for (let i = 0; i < len; i++) {
      let cur = q.shift()
      if (cur.left) q.push(cur.left)
      if (cur.right) q.push(cur.right)
    }
    if (q[q.length - 1]) res.push(q[q.length - 1].val)
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
    grid[x][y] = '0'
    let directions = [-1, 0, 1, 0, -1]
    for (let i = 0; i < directions.length - 1; i++) {
      let newX = x + directions[i]
      let newY = y + directions[i + 1]
      if (!outOfBound(newX, newY) && grid[newX][newY] == 1) dfs(newX, newY)
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
## 209. Minimum Size Subarray Sum  
[link](https://leetcode.com/problems/minimum-size-subarray-sum/)  
```javascript
const minSubArrayLen = function(s, nums) {
  let lo = 0, hi = 0
  const len = nums.length
  let min = Infinity, sum = 0
  while (hi < len) {
    sum += nums[hi++]
    while (sum >= s) {
      min = Math.min(min, hi - lo)
      sum -= nums[lo++]
    }
  }
  return min == Infinity ? 0 : min
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
    let i = lo, j = hi + 1   // ATT
    while (true) {
      // find the nums[i] more than nums[lo]
      while (i < hi && nums[++i] < nums[lo]) {}
      // find the nums[j] less than nums[lo]
      while (j > lo && nums[lo] < nums[--j]) {}
      // if i is more than j, then no swap
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
## 221. Maximal Square  
[link](https://leetcode.com/problems/maximal-square/)  
```javascript
const maximalSquare = function(matrix) {
  if (!matrix.length || !matrix[0].length) return 0
  const dp = [...new Array(matrix.length + 1)].map(_ => new Array(matrix[0].length + 1).fill(0))
  let max = 0
  for (let i = 1; i < dp.length; i++) {
    for (let j = 1; j < dp[0].length; j++) {
      if (matrix[i - 1][j - 1] == 1) {
        dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        max = Math.max(dp[i][j], max)
      }
    }
  }
  return max * max
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
  let parent = new Map()
  let stack = []
  parent.set(root, null)
  stack.push(root)
  
  while (!parent.has(p) || !parent.has(q)) {
    let node = stack.pop()
    if (node.left) {
      parent.set(node.left, node)
      stack.push(node.left)
    }
    if (node.right) {
      parent.set(node.right, node)
      stack.push(node.right)
    }
  }
  let ancestor = new Set()
  while (p != null) {
    ancestor.add(p)
    p = parent.get(p)
  }
  while (!ancestor.has(q)) {
    q = parent.get(q)
  }
  return q
  // dfs
  if (!root || root == p || root == q) return root
  let left = lowestCommonAncestor(root.left, p, q)
  let right = lowestCommonAncestor(root.right, p, q)
  return (left && right) ? root : (left || right)
}
```
## 264. Ugly Number II  
[link](https://leetcode.com/problems/ugly-number-ii/)  
```javascript
const nthUglyNumber = function(n) {
  if (n <= 0) return
  let dp = Array(n).fill(0)
  let pt2 = 0, pt3 = 0, pt5 = 0
  dp[0] = 1
  for (let i = 1; i < n; i++) {
    dp[i] = Math.min(dp[pt2] * 2, dp[pt3] * 3, dp[pt5] * 5)
    if (dp[i] === dp[pt2] * 2) pt2++
    if (dp[i] === dp[pt3] * 3) pt3++
    if (dp[i] === dp[pt5] * 5) pt5++
  }
  return dp[n - 1]
}
```
## 283. Move Zeros  
[link](https://leetcode.com/problems/move-zeroes/)  
```javascript
const moveZeroes = function(nums) {
  let idx = 0  // index of zeros(sometimes may the same then idx++)
  // one pass find the non-zero and swap with zero , then idx++
  for (let i = 0; i < nums.length; i++) {
    if (nums[i] !== 0) {
      nums[idx] = nums[i]
      nums[i] = idx === i ? nums[i] : 0
      idx++
    }
  }
}
```
## 286. Walls and Gates  
[link](https://leetcode.com/problems/walls-and-gates/)  
```javascript
const wallsAndGates = function(rooms) {
  if (!rooms || !rooms.length || !rooms[0].length) return
  let q = []
  for (let i = 0; i < rooms.length; i++) {
    for (let j = 0; j < rooms[0].length; j++) {
      if (rooms[i][j] == 0) q.push([i, j])
    }
  }
  while (q.length) {
    let [x, y] = q.shift()
    let directions = [-1, 0, 1, -1, 0]
    for (let i = 0; i < directions.length - 1; i++) {
      let newX = x + directions[i]
      let newY = y + directions[i + 1]
      if (!outOfBound(newX, newY) && rooms[newX][newY] == 2147483647) {
        rooms[newX][newY] = rooms[x][y] + 1
        q.push([newX, newY])
      }
    }
  }
  function outOfBound(x, y) {
    return x < 0 || y < 0 || x >= rooms.length || y >= rooms[0].length
  }
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
    let cur = data.shift()
    if (cur == null) return null
    else {
      let node = new TreeNode(cur)
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
## 315. Count of Smaller Numbers After Self  
[link](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)  
```javascript
const countSmaller = function(nums) {
  let arr = []
  let res = new Array(nums.length).fill(0)

  for (let i = nums.length - 1; i >= 0; i--) {
    let lo = 0, hi = arr.length
    while (lo < hi) {
      let mid = lo + ((hi - lo) >> 1)
      if (arr[mid] < nums[i]) lo = mid + 1
      else hi = mid
    }
    res[i] = lo
    arr.splice(lo, 0, nums[i])
  }
  return res
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
  // map will be in order 
  // map[num] = i means num shows i times
  // then buckets' indexes are the apperance times
  for (let num in map) {
    bucket[map[num]].push(num | 0)
  }
  // bucket[i] = [num] means num show i times
  for (let i = nums.length; i >= 0 && k > 0; k--) {
    while (bucket[i].length === 0) i--
    res.push(bucket[i].pop())
  }
  return res
}
// O(N)
```
## 394. Decode String  
[link](https://leetcode.com/problems/decode-string/)  
```javascript
const decodeString = function(s) {
  const stack = [[1, '']]
  for (let i = 0; i < s.length; i++) {
    if (isDigit(s[i])) {
      const numStr = parseNumber(s, i)
      stack.push([parseInt(numStr), ''])
      i += numStr.length - 1
    } else if (s[i] === '[') {
      continue
    } else if (s[i] === ']') {
      const [nRepeats, str] = stack.pop()
      const top = stack[stack.length - 1]
      top[1] += str.repeat(nRepeats)
    } else {
      const top = stack[stack.length - 1]
      top[1] += s[i]
    }
  }
  const [nRepeats, str] = stack.pop()
  return str
  
  function isDigit(c) { return /[0-9]/.test(c) }
  function parseNumber(s, start) {
    let i = start
    while (isDigit(s[i])) { i++ }
    return s.substring(start, i)
  }
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
## 438. Find All Anagrams in a String  
[link](https://leetcode.com/problems/find-all-anagrams-in-a-string/)  
```javascript
const findAnagrams = function(s, p) {
  let unique = 0
  let map = new Map()
  for (let c of p) {
    if (map[c] == null) {
      unique++
      map[c] = 1
    } else map[c]++
  }
  let lo = 0, hi = 0
  const res = []
  for (hi; hi < s.length; hi++) {
    if (map[s[hi]] != null) map[s[hi]]--
    if (map[s[hi]] == 0) unique--
    if (unique === 0) res.push(lo)
    if (hi - lo + 1 === p.length) {
      if (map[s[lo]] != null) map[s[lo]]++
      if (map[s[lo++]] === 1) unique++
    }
  }
  return res
}
```
## 442. Find All Duplicates in an Array  
[link](https://leetcode.com/problems/find-all-duplicates-in-an-array/)  
```javascript
const findDuplicates = function(nums) {
  const res = []
  for (let i = 0; i < nums.length; i++) {
    let idx = Math.abs(nums[i]) - 1
    // set num appeared to negative
    // when again < 0 means duplicated
    if (nums[idx] < 0) res.push(idx + 1)
    nums[idx] = -nums[idx]
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
  let cur = null
  let carry = 0
  while (s1.length || s2.length || carry) {
    carry += s1.length ? s1.pop() : 0
    carry += s2.length ? s2.pop() : 0
    cur = new ListNode(carry % 10)
    carry = carry / 10 | 0
     // just insert the new cur between dummy and next
    cur.next = dummy.next
    dummy.next = cur
  }
  return dummy.next

}
```
## 448. Find All Numbers Disappeared in an Array  
[link](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)  
```javascript
const findDisappearedNumbers = function(nums) {
  const res = []
  for (let i = 0; i < nums.length; i++) {
    let m = Math.abs(nums[i]) - 1
    // meet the same do nothing, still negative
    // then the num[idx] will be positive
    nums[m] = nums[m] > 0 ? -nums[m] : nums[m]
  }
  for (let i = 0; i < nums.length; i++) {
    if (nums[i] > 0) res.push(i + 1)
  }
  return res
}
```
## 460. LFU Cache  
[link](https://leetcode.com/problems/lfu-cache/)  
```javascript
class Node {
  constructor(key, value) {
    this.key = key
    this.val = value
    this.next = this.prev = null
    this.freq = 1
  }
}

class DoublyLinkedList {
  contructor() {
    this.head = new Node(null, null)
    this.tail = new Node(null, null)
    this.head.next = this.tail
    this.tail.prev = this.head
  }

  insertHead(node) {
    node.prev = this.head
    node.next = this.head.next
    this.head.next.prev = node
    this.head.next = node
  }
  removeNode(node) {
    node.prev.next = node.next
    node.next.prev = node.prev
  }
  removeTail() {
    let tail = this.tail.prev
    this.removeNode(tail)
    return tail.key
  }
  isEmpty() {
    return this.head.next.val == null
  }
}
let LFUCache = function(capacity) {
  this.capacity = capacity
  this.currentSize = 0
  this.leastFreq = 0
  this.nodeHash = new Map()
  this.freqHash = new Map()
}

LFUCache.prototype.get = function(key) {
  let node = this.nodeHash.get(key)
  if (!node) return -1
  // remove node in  freq list
  this.freqHash.get(node.freq).removeNode(node)
  // check whether it's least freq and are there nodes left 
  // if none, leastFreq++ 
  if (node.freq == this.leastFreq && this.freqHash.get(node.freq).isEmpty()) this.leastFreq++
  node.freq++
  // check whether there're nodes that are the same with the new freq
  // if none, new a list
  if (this.freqHash.get(node.freq) == null) this.freqHash.set(node.freq, new DoublyLinkedList())
  // insert Head
  this.freqHash.get(node.freq).insertHead(node)
  return node.val
}

LFUCache.prototype.put = function(key, value) {
  if (!this.capacity) return 
  let node = this.nodeHash.get(key)
  if (!node) { // new node
    this.currentSize++
    if (this.currentSize > this.capacity) {
      let tailKey = this.freqHash.get(this.leastFreq).removeTail()
      this.nodeHash.delete(tailKey)
      this.currentSize--
    }
    let newNode = new Node(key, value)
    if (this.freqHash.get(1) == null) this.freqHash.set(1, new DoublyLinkedLlist())
    this.freqHash.get(1).insertHead(newNode)
    this.nodeHash.set(key, newNode)
    this.leastFreq = 1
  } else { // existed
    node.val = value
    this.freqHash.get(node.freq).removeNode(key)
    if (node.freq === this.leastFreq && this.freqHash.get(node.freq).isEmpty()) this.leastFreq++
    node.freq++
    if (this.freqHash.get(node.freq) == null) this.freqHash.set(node.freq, new DoublyLinkedList())
    this.freqHash.set(node.freq).insertHead(node)
  }
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
## 494. Target Sum //TODO  
[link](https://leetcode.com/problems/target-sum/)  
```javascript
const findTargetSumWays = function(nums, S) {
  let sum = 0 
  for (let num of nums) {sum += num}
  if (sum < S || -sum > S) return 0
  const dp = Array(2 * sum + 1).fill(0)
  dp[0 + sum] = 1
  for (let i = 0; i < nums.length; i++) {
    let next = new Array(2 * sum + 1).fill(0)
    for (let k = 0; k < 2 * sum + 1; k++) {
      if (dp[k] != 0) {
        next[k + nums[i]] += dp[k]
        next[k - nums[i]] += dp[k]
      }
    }
    dp = next
  }
  return dp[sum + S]
}
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

  // memo
  let first = 0
  let second = 1
  while (N > 2) {
    [first, second] = [second, first + second]
    N--
  }
  return N ? first + second : 0
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
  if (!root) return 0
  let max = 0
  dpt(root)
  return max
  
  function dpt(node) {
    if (!node) return 0
    let l = dpt(node.left)
    let r = dpt(node.right)
    
    max = Math.max(l + r, max)
    return 1 + Math.max(l, r)
  }
}
```
## 581. Shortest Unsorted Continuous Subarray  
[link](https://leetcode.com/problems/shortest-unsorted-continuous-subarray/)  
```javascript
const findUnsortedSubarray = function(nums) {
  let n = nums.length
  // hi means the max index (after that is in right order)
  // lo means the min index (before that is in right order)
  let lo = -1, hi = -2
  let min = nums[n - 1], max = nums[0]
  for (let i = 1; i < n; i++) {
    max = Math.max(nums[i], max)
    min = Math.min(nums[n - i - 1], min)
    // if nums[i] is less than max, means after the previous max
    // there's a unorder element, then the hi moves to i
    if (nums[i] < max) hi = i
    if (nums[n - i - 1] > min) lo = n - i - 1
  }
  return hi - lo + 1
}
```
## 617. Merge Two Binary Trees  
[link](https://leetcode.com/problems/merge-two-binary-trees/)  
```javascript
const mergeTrees = function(t1, t2) {
  // bfs
  if (!t1 || !t2) return t1 || t2
  let stack = []
  stack.push([t1, t2])
  
  while (stack.length) {
    let [n1, n2] = stack.shift()
    if (!n1 || !n2) continue
    n1.val += n2.val
    if (!n1.left) n1.left = n2.left
    else stack.push([n1.left, n2.left])
    if (!n1.right) n1.right = n2.right
    else stack.push([n1.right, n2.right])
  }
  return t1
  // dfs
  if (!t1 && !t2) return null
  let val = (t1 ? t1.val : 0) + (t2 ? t2.val : 0)
  let node = new TreeNode(val)
  node.left = mergeTrees(t1 && t1.left, t2 && t2.left)
  node.right = mergeTrees(t1 && t1.right, t2 && t2.right)
  return node
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
## 739. Daily Temperatures  
[link](https://leetcode.com/problems/daily-temperatures/)  
```javascript
const dailyTemperatures = function(T) {
  let stack = []
  let res = new Array(T.length).fill(0)
  for (let i = 0; i < T.length; i++) {
    while (stack.length && T[i] > T[stack[stack.length - 1]]) {
      let idx = stack.pop()
      res[idx] = i - idx
    }
    stack.push(i)
  }
  return res
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
## 842. Split Array into Fibonacci Sequence  
[link](https://leetcode.com/problems/split-array-into-fibonacci-sequence/)  
```javascript
const splitIntoFibonacci = function(S) {
  const res = []
  helper(0, [])
  return res

  function helper(idx, tmp) {
    if (idx === S.length && tmp.length > 2) {
      res = tmp.slice()
      return
    } else {
      for (let i = idx; i < S.length; i++) {
        if (S[idx] === '0' && i != idx) return
        let num = S.slice(idx, i + 1) | 0
        if (num > Math.pow(2, 31) - 1) return
        if (tmp.length < 2 || tmp[tmp.length - 2] + tmp[tmp.length - 1] == num) {
          // choose
          tmp.push(num)
          // explore
          helper(i + 1, tmp)
          // un-choose
          tmp.pop()
        }
      }
    }
  }
}
```
## 876. Middle of the Linked List  
[link](https://leetcode.com/problems/middle-of-the-linked-list/)  
```javascript
const middleNode = function(head) {
  let fast = head
  let slow = head
  while (fast && fast.next) {
    fast = fast.next.next
    slow = slow.next
  }
  return slow
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
      q.push(cur.left)  // no need worry about the null
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
        if (!outOfBound(newX, newY) && grid[newX][newY] == 1) {
          grid[newX][newY] = 2
          fresh--
          next.push([newX, newY])
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

