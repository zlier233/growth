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
#### Combination Sum
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
#### Combination Sum II  
> Given a collection of candidate numbers (candidates) and a target number (target),   
> find all unique combinations in candidates where the candidate numbers sums to target.  
> Each number in candidates may only be used once in the combination.  
``` javascript
const combinationSum2 = function(candidates, target) {
  const res = []
  candidates.sort((a, b) => a - b)
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
        helper(i + 1, left - candidates[i], tmp)
        // un-choose
        tmp.pop()
        // remove duplicates
        while (candidates[i] === candidates[i + 1]) {i++}
      }
    }
  }
}
```
#### Combination Sum III
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
    if (left < 0) return 
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
#### Subsets I && II 
> Given a set of distinct integers, nums, return all possible subsets (the power set).  
```javascript 
const subsets = function(nums) {
  const res = []
  nums.sort((a, b) => a - b) // if has duplicates
  helper(0, [])
  return res

  function helper(idx, tmp) {
    res.push(tmp.slice)
    for (let i = idx; i < nums.length; i++) {
      // choose
      tmp.push(nums[i])
      // explore
      helper(i + 1, tmp)
      // un-choose
      tmp.pop()
      // remove duplicates
      while (nums[i] === nums[i + 1]) {i++}
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
        // un-choose
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
#### Sudoku Solver  
> As description 37.
``` javascript
const solveSudoku = function(board) {
  helper(0, 0)

  function helper(i, j) {
    if (i === 9) return true
    if (j === 9) return helper(i + 1, 0)
    if (board[i][j] != '.') return helper(i, j + 1)

    for (let c = 1; c <= 9; c++) {
      if (check(i, j, c)) {
        // choose
        board[i][j] = String(c)
        // explore
        if (helper(i, j + 1)) return true
        // un-choose
        board[i][j] = '.'
      }
      return false
    }
    
    function check(i, j, val) {
      for (let k = 0; k < 9; k++) {
        // check col 
        if (board[k][j] == val) return false
        // check row
        if (board[i][k] == val) return false
        // check cube
        if (board[i - i % 3 + k / 3 | 0][j - j % 3 + k % 3] == val) return false
      }

      return true
    }
  }
}
```
#### Word Search  
> As description 79.
```javascript  
const exist = function(board, word) {
  for (let i = 0; i < board.length; i++) {
    for (let j = 0; j < board[0].length; j++) {
      if (helper(i, j, board, 0)) return true
    }
  }
  return false

  function helper(idx, idy, board, path) {
    if (idx < 0 || idy < 0 || idx >= board.length || idy >= board[0].length || 
      board[idx][idy] !== word[path] || path > word.length) return false

    // choose
    board[idx][idy] = '*'
    path++
    // judge
    if (path === word.length) return true
    // explore
    let isFound = helper(idx + 1, idy, board, path) ||
                  helper(idx - 1, idy, board, path) ||
                  helper(idx, idy + 1, board, path) ||
                  helper(idx, idy - 1, board, path)
    // un-choose
    board[idx][idy] = word[--path]
  }
  return isFound
}
```
#### Word Search II  
> As description 212
```javascript
// use trie
const findWords = function(board, words) {
  let res = []
  const root = buildTrie()
  for (let i = 0; i < board.length; i++) {
    for (let j = 0; j < board[0].length; j++) {
      search(root, i, j)
    }
  }
  return res
  
  function buildTrie() {
    const root = {}
    for (let w of words) {
      let node = root
      for (let c of w) {
        if (node[c] == null) node[c] = {}
        node = node[c]
      }
      node.word = w
    }
    return root
  }
  
  function search(node, i, j) {
    if (node.word != null) {
      res.push(node.word)
      node.word = null
    }
    if (i < 0 || j < 0 || i >= board.length || j >= board[0].length) return
    if (node[board[i][j]] == null) return 
    
    // choose
    const c = board[i][j]
    board[i][j] = '#'
    // explore
    search(node[c], i + 1, j)
    search(node[c], i - 1, j)
    search(node[c], i, j + 1)
    search(node[c], i, j - 1)
    // un-choose
    board[i][j] = c
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
#### Regular Expression Matching  
> Like regexp  
```javascript
// actually using dp =.=
const isMatch = function(s, p) {
  const lenS = s.length, lenP = p.length
  const map = new Map()
  return check(0, 0)

  function check(ids, idp) {
    if (map[ids + ':' + idp] != undefined) return map[ids + ':' + idp]
    if (ids > lenS) return false
    if (ids === lenS && idp === lenP) return true

    // p: ?.? / ?a?  s: ?a?
    if (p[idp] === '.' || p[idp] === s[ids]) {
      map[ids + ':' + idp] = p[idp + 1] === '*' ?
        check(ids + 1, idp) || check(ids, idp + 2) :
        check(ids + 1, idp + 1)
    } else {
      map[ids + ':' + idp] = p[idp + 1] === '*' ?
        check(ids, idp + 2) : false 
    }
    return map[ids + ':' + idp]
  }
}
```
#### Restore IP Address  
> As description 93  
``` javascript
const restoreIpAddress = function(s) {
  const res = []
  helper([], 0)
  return res

  function helper(tmp, idx) {
    if (tmp.length === 4 && idx < s.length) return
    else if (tmp.length === 4 && idx === s.length) res.push(tmp.slice().join('.'))
    else {
      for (let i = idx; i < s.length; i++) {
        if (i != idx && s[i] === '0') return
        const num = parseInt(s.slice(idx, i + 1))
        if (num > 255) return
        // choose
        tmp.push(num)
        // explore
        helper(tmp, i + 1)
        // un-choose
        tmp.pop()
      }
    }
  }
}
```
#### Palindrome Partitioning  
> 131  
``` javascript
const partition = function(s) {
  const res = []
  helper(0, []) 
  return res

  function helper(idx, tmp) {
    if (tmp.length > 0 && idx >= s.length) res.push(tmp.slice())
    else {
      for (let i = idx; i < s.length; i++) {
        if (isPalindrome(idx, i)) {
          // choose
          tmp.push(s.slice(idx, i + 1))
          // explore
          helper(i + 1, tmp)
          // un-choose
          tmp.pop()
        }
      }
    }
  }

  function isPalindrome(start, end) {
    if (start === end) return true
    while (start < end) {
      if (s[start] !== s[end]) return false
      else {start++;end--}
    }
    return true
  }
}
```

### TIME-COMPLEXITY  
According to the specific problems. O(N!)/ O(2^N) / balabala
## Binary Search  
### STRUCTURE CODE  
``` javascript
// NORMAL  
const bs = function(nums, target) {
  if (!nums.length) return -1
  let lo = 0, hi = nums.length - 1
  while (lo <= hi) {
    let mid = lo + ((hi - lo) >> 1)
    if (nums[mid] === target) return mid
    else if (nums[mid] > target) hi = mid - 1
    else lo = mid + 1
  }
  return -1
}
// LEFT BOUND
const bs = function(nums, target) {
  if (!nums.length) return -1
  let lo = 0, hi = nums.length
  while (lo < hi) {
    let mid = lo + ((hi - lo) >> 1)
    if (nums[mid] === target) hi = mid
    else if (nums[mid] > target) hi = mid
    else lo = mid + 1
  }
  return lo
}
// RIGHT BOUND
const bs = function(nums, target) {
  if (!nums.length) return -1
  let lo = 0, hi = nums.length
  while (lo < hi) {
    let mid = lo + ((hi - lo) >> 1)
    if (nums[mid] === target) lo = mid + 1
    else if (nums[mid] < target) lo = mid + 1
    else hi = mid
  }
  return lo - 1
}
```
### EXAMPLES  
#### Median of Two Sorted Array  
> 4  
``` javascript
const findMedianSortedArrays = function(nums1, nums2) {
  const m = nums1.length, n = nums2.length, resLeft = 0, resRight = 0
  // swap
  if (m > n) {
    let tmp = n, tmpNums = num2
    n = m
    nums2 = nums1
    m = tmp
    nums1 = tmpNums
  }
  let lo = 0, hi = m, halfLen = (m + n + 1) >> 1
  while (lo <= hi) {
    let i = lo + ((hi - lo) >> 1), j = halfLen - i
    if (i < m && nums2[j - 1] > nums1[i]) lo = i + 1
    else if (i > 0 && nums1[i - 1] > nums2[j]) hi = i - 1
    else {
      if (i === 0) resLeft = nums2[j - 1]
      else if (j === 0) resLeft = nums1[i - 1]
      else resLeft = Math.max(nums2[j - 1], nums1[i - 1])

      if ((m + n) % 2 === 1) return resLeft

      if (i === m) resRight = nums2[j]
      else if (j === n) resRight = nums1[i]
      else resRight = Math.min(nums1[i], nums2[j])

      return (resLeft + resRight) / 2.0
    }
  }
}
```
#### Search In Sorted Array I && II
> 33, 81  
``` javascript
const search = function(nums, target) {
  let lo = 0, hi = nums.length - 1
  while (lo <= hi) {
    let mid = lo + ((hi - lo) >> 1)
    if (target === nums[mid]) return mid   // true for 81
    // remove duplicates in right part for 81
    while (nums[mid] === nums[hi] && mid !== hi) { hi-- }
    if (nums[mid] > nums[hi]) {
      if (target < nums[mid] && target >= nums[lo]) hi = mid - 1
      else lo = mid + 1
    } else {
      if (target > nums[mid] && target <= nums[hi]) lo = mid + 1
      else hi = mid - 1
    }
  }
  return -1  // false for 81
}
```
#### Search a 2D Matrix  
> 74
``` javascript
const searchMatrix = function(matrix, target) {
  if (!matrix || !matrix.length || !matrix[0].length) return false
  let lo = 0, hi = matrix.length * matrix[0].length - 1
  while (lo <= hi) {
    let mid = lo + ((hi - lo) >> 1)
    let x = mid % matrix[0].length
    let y = mid / matrix[0].length | 0
    if (target === matrix[x][y]) return true
    else if (target > matrix[x][y]) lo = mid + 1
    else hi = mid - 1
  }
  return false
}
```
#### Search a 2D Matrix II  
> 240. Top Right is max, right to left, up to down, decreasing.  
```javascript
const searchMatrix = function(matrix, target) {
  if (!matrix || !matrix.length || !matrix[0].length) return false
  let x = 0, y = matrix[0].length - 1
  while (x < matrix.length && y >= 0) {
    if (matrix[x][y] === target) return true
    else if (target > matrix[x][y]) x++
    else y--
  }
  return false
}
```
#### Search Insert Position  
> 35
``` javascript
// left bound
const searchInsert = function(nums, target) {
  let lo = 0, hi = nums.length
  while (lo < hi) {
    let mid = lo + ((hi - lo) >> 1)
    if (target > nums[mid]) lo = mid + 1
    else hi = mid
  }
  return lo
}
```
#### Kth Smallest Element in a Sorted Matrix  
> 378  
``` javascript
// The correctness of this algorithm is to ensure that the target value is   
// within the range of [low, high] for each loop step.
const kthSmallest = function(matrix, k) {
  let lo = matrix[0][0], N = matrix.length, hi = matrix[N - 1][N - 1]
  while (lo < hi) {
    let mid = lo + ((hi - lo) >> 1)
    let cnt = 0
    for (let i = 0; i < N; i++) {
      let j = N - 1
      // count num smaller than mid
      while (matrix[i][j] > mid && j >= 0) { j-- }  
      cnt += j + 1
    }
    if (cnt < k) lo = mid + 1
    else hi = mid
  }
  return lo
}
```
#### Find Peak Element  
> 162
``` javascript
const findPeakElement = function(nums) {
  let lo = 0, hi = nums.length - 1
  while (lo < hi) {
    let mid = lo + ((hi - lo) >> 1)
    // make each boundary hold true
    if (nums[mid] < nums[mid + 1]) lo = mid + 1
    else hi = mid
  }
  return lo
}
```
#### Find the Duplicate Number  
> 287
``` javascript
const findDuplicate = function(nums) {
  let lo = 1, hi = nums.length
  while (lo < hi) {
    let mid = lo + ((hi - lo) >> 1)
    let cnt = 0
    for (let j = 0; j < nums.length; j++) {
      if (nums[j] <= mid) cnt++
    }
    if (cnt > mid) hi = mid // duplicates in [lo, mid]
    else lo = mid + 1 // duplicates in [mid + 1, hi]
  }
}
```
#### Pow(x, n)  
> 50
```javascript
const myPow = function(x, n) {
  if (n === 0) return 1
  if (n < 0) return 1 / myPow(x, -n)
  if (n & 1) return x * myPow(x, n - 1) 
  return myPow(x * x, n / 2)
}
```
#### Sqrt(x)  
> 69
```javascript
const mySqrt = function(x) {
  let hi = x
  while (hi * hi > x) {
    hi = (hi + x / hi) / 2 | 0
  }
  return hi
}
const mySqrt = function(x) {
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
#### Find Minimum in Rotated Sorted Array  
> 153
``` javascript
const findMin = function(nums) {
  let lo = 0, hi = nums.length - 1
  while (lo < hi) {
    let mid = lo + ((hi - lo) >> 1)
    if (nums[mid] > nums[hi]) lo = mid + 1
    else hi = mid
  }
  return nums[lo]
}
```
#### Find Minimum in Rotated Sorted Array II  
> 154
``` javascript
const findMin = function(nums) {
  let lo = 0, hi = nums.length - 1
  while (lo < hi) {
    let mid = lo + ((hi - lo) >> 1)
    if (nums[mid] > nums[hi]) lo = mid + 1
    else if (nums[mid] < nums[hi]) hi = mid
    else {
      if (nums[hi - 1] > nums[hi]) {
        lo = hi
        break
      }
      // nums[mid] == nums[hi] shrink the upper bound like 81
      hi--
    }
  }
  return nums[lo]
}
```
#### Longest Increasing Subsequence  
> 300 
``` javascript
const lengthOfLIS = function(nums) {
  let tails = Array(nums.length).fill(0)
  let max = 0
  for (let n of nums) {
    let i = 0, j = max
    // search for the pos of num in tails
    while (i < j) {
      let mid = i + ((j - i) >> 1)
      if (num > tails[mid]) i = mid + 1
      else j = mid
    }
    // update tails, will cover previous bigger one
    tails[i] = num
    // if insert to the last, then max++
    if (max === i) max++
  }
  return max
}
```
#### Count Of Smaller Numbers After Self  
> 315  
```javascript
// a little bit similar to 300
const countSmaller = function(nums) {
  const len = nums.length
  const res = Array(len).fill(0)
  const arr = []
  for (let i = len - 1; i >= 0; i--) {
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
#### Koko Eating Bananas  
> 875  
```javascript
const minEatingSpeed = function(piles, H) {
  let lo = 1, hi = Math.max(...piles) + 1
  while (lo < hi) {
    let mid = lo + ((hi - lo) >> 1)
    let cnt = 0
    for (let i = 0; i < piles.length; i++) {
      cnt += (piles[i] + mid - 1) / mid | 0
    }
    if (cnt <= H) hi = mid
    else lo = mid + 1
  }
  return lo
}
```
#### Capacity To Ship Packages Within D Days  
> 1011 
```javascript
const shipWithinDays = function(weights, D) {
  let sum = 0
  for (let w of weights) {sum += w}
  let lo = Math.max(...weights), hi = sum + 1
  while (lo < hi) {
    let mid = lo + ((hi - lo) >> 1)
    let cnt = 1
    let tmp = 0
    for (let w of weights) {
      if (tmp + w > mid) {
        cnt++
        tmp = 0
      }
      tmp += w
    }
    if (cnt > D) lo = mid + 1
    else hi = mid
  }
  return lo
}
```

## BFS & DFS 
### STRUCTURE CODE  
```javascript
function bfs(start, target) {
  let q = [start]
  let visited = new Set([start])
  let cnt = 0

  while (q.length) {
    const len = q.length
    for (let i = 0; i < len; i++) {
      let cur = q.shift()
      // judge
      if (q === target) return cnt
      // add neighbors to q
      for (let node of q) {
        if (!visited.has(node)) {
          q.push(node)
          visited.add(node)
        }
      }
    }
    cnt++
  }

  function dfs() {
    /* It's similar to the backtracking but not exactly the same. */
  }
}
```
### EXAMPLES  
#### Binary Tree Level Order Traversal  
> Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).  
``` javascript
// bfs
const levelOrder = function(root) {
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

  // dfs
  if (!root) return []
  const res = []
  dfs(root, 0)
  return res

  function dfs(node, l) {
    if (!node) return
    if (!res[l]) res[l] = []
    res[l].push(node.val)
    if (node.left) dfs(node.left, l + 1)
    if (node.right) dfs(node.right, l + 1)
  }
}
```
#### Binary Tree Level Order Traversal II  
> Given a binary tree, return the bottom-up level order traversal of its nodes' values. (ie, from left to right,   
> level by level from leaf to root).
```javascript
// bfs
const levelOrderBottom = function(root) {
  if (!root) return []
  let q = [root], cnt = 1
  let res = []
  const maxl = dpt(root)

  while (q.length) {
    const len = q.length
    let tmp = []
    for (let i = 0; i < len; i++) {
      let cur = q.shift()
      tmp.push(cur.val)
      if (cur.left) q.push(cur.left)
      if (cur.right) q.push(cur.right)
    }
    res[maxl - cnt] = tmp
    cnt++
  }
  return res

  function dpt(root) {
    if (!root) return 0
    return 1 + Math.max(dpt(root.left), dpt(root.right))
  }
}
```
#### Binary Tree Zigzag Level Order Traversal  
> Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right,   
> then right to left for the next level and alternate between).  
```javascript
// bfs
const zigzagLevelOrder = function(root) {
  if (!root) return []
  let q = [root]
  let res = []
  let zigzag = true

  while (q.length) {
    const len = q.length
    let tmp = [], nxt = []
    for (let i = 0; i < len; i++) {
      let cur = q.pop()   // ATTENTION
      tmp.push(cur.val)
      if (zigzag) {
        if (cur.left) nxt.push(cur.left)
        if (cur.right) nxt.push(cur.right)
      } else {
        if (cur.right) nxt.push(cur.right)
        if (cur.left) nxt.push(cur.left)
      }
    }
    res.push(tmp)
    zigzag = !zigzag
    q = nxt
  }
  return res
}

```
#### Binary Tree Vertical Order Traversal  
> Given a binary tree, return the vertical order traversal of its nodes' values.   
> (ie, from top to bottom, column by column).
```javascript
// bfs
const vericalOrder = function(root) {
  if (!root) return []
  let res = [], min = 0
  let cols = new Map()  // store the cols value of nodes
  let map = new Map()   // store the col-nodes array
  let q = [root]
  cols.set(root, 0)
  while (q.length) {
    let cur = q.shift()
    let col = cols.get(cur)
    if (!map.has(col)) map.set(col, [])
    map.get(col).push(cur.val)  // push the cur node to its col's map
    if (cur.left) {
      q.push(cur.left)
      cols.set(cur.left, col - 1)
    }
    if (cur.right) {
      q.push(cur.right)
      cols.set(cur.right, col + 1)
    }
    min = Math.min(min, col)
  }
  while (map.has(min++)) {
    res.push(map.get(min))
  }
  return res
}

```
#### N-ary Tree Level Order Traversal  
> Given an n-ary tree, return the level order traversal of its nodes' values.  
```javascript
// bfs
const levelOrder = function(root) {
  if (!root) return []
  let q = [root]
  let res = []
  while (q.length) {
    const len = q.length
    let tmp = []
    for (let i = 0; i < len; i++) {
      let cur = q.shift()
      tmp.push(cur.val)
      for (let n of cur.children) {
        q.push(n)
      }
    }
    res.push(tmp)
  }
  return res
}
```

#### Number of Islands  
> As description  
```javascript
// bfs
const numIslands = function(grid) {
  if (!grid || !grid.length || !grid[0].length) return 0
  const m = grid.length, n = grid[0].length
  let cnt = 0
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      if (grid[i][j] === '1') {
        bfs(i, j)
        cnt++
      }
    }
  }
  return cnt

  function bfs(x, y) {
    let q = [[x, y]]
    grid[x][y] = '0'
    const directions = [-1, 0, 1, 0, -1]
    while (q.length) {
      let [x, y] = q.shift()
      for (let i = 0; i < directions.length - 1; i++) {
        let newX = x + directions[i]
        let newY = y + directions[i + 1]
        if (!outOfBound(newX, newY) && grid[newX][newY] === '1') {
          q.push([newX, newY])
          grid[newX][newY] = '0'
        }
      }
    }
  }
  function outOfBound(x, y) {
    return x < 0 || y < 0 || x >= m || y >= n
  }

  // dfs
  if (!grid || !grid.length || !grid[0].length) return 0
  const m = grid.length, n = grid[0].length
  let cnt = 0
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      dfs(i, j)
      cnt++
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

#### Max Area Of Island  
> As description
```javascript
const maxAreaOfIsland = function(grid) {
  // bfs
  let m = grid.length, n = grid[0].length
  let max = 0
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      if (grid[i][j] === '1') {
        grid[i][j] = '0'
        let sum = bfs(i, j)
        max = Math.max(sum, max)
      }
    }
  }
  return max

  function bfs(x, y) {
    let q = [[x, y]]
    let sum = 0
    const directions = [-1, 0, 1, 0, -1]
    while (q.length) {
      let [x, y] = q.shift()
      sum++

      for (let i = 0; i < directions.length - 1; i++) {
        let newX = x + directions[i]
        let newY = y + directions[i + 1]

        if (!outOfBound(newX, newY) && grid[newX, newY] === '1') {
          grid[newX][newY] = '0'
          q.push([newX, newY])
        }
      }
    }
    return sum
  }
  function outOfBound(x, y) {
    return x < 0 || y < 0 || x >= m || y >= n
  }

  // dfs
  if (!grid || !grid.length || !grid[0].length) return 0
  const m = grid.length, n = grid[0].length
  let max = 0
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      if (grid[i][j] == 1) {
        max = Math.max(dfs(i, j), max)
      }
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
}
```
#### Walls And Gates  
> As description  
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
    let directions = [-1, 0, 1, 0, -1]
    
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
#### Word Ladder  
> As description  
```javascript 
const ladderLength = function(beginWord, endWord, wordList) {
  let q = [beginWord]
  const dict = new Set(wordList)
  const seen = new Set([beginWord])
  let cnt = 1
  while (q.length) {
    const len = q.length
    for (let i = 0; i < len; i++) {
      let cur = q.shift()
      if (cur === endWord) return cnt
      
      let arr = cur.split('')
      for (let i = 0; i < arr.length; i++) {
        for (let d = 0; d < 26; d++) {
          arr[i] = String.fromCharCode(97 + d)
          const nv = arr.join('')
          if (!seen.has(nv) && dict.has(nv)) {
            seen.add(nv)
            q.push(nv)
          }
          arr[i] = cur[i]
        }
      }
    }
    cnt++
  }
  return 0
}
```


#### Critical Connections in a Network  
> As description 1192.  
```javascript
const criticalConnections = function(n, connections) {
  // build graph
  const g = Array(n).fill([])
  for (let [u, v] of connections) {
    g[u].push(v)
    g[v].push(u)
  }
  let idx = 0
  const res = 0
  const low = 0
  const dfn = Array(n).fill(Infinity)
  dfs(0, -1)
  return res

  function dfs(u, pre) {
    low[n] = dfn[u] = idx++
    for (const v of g[u]) { // scan
      if (v === pre) continue  // parent vertex, ignore
      if (dfn[v] === Infinity) {  // v is not visited yet
        dfs(v, u)
        low[u] = Math.min(low[u], low[v])
        if (low[v] > dfn[u]) res.push([u, v]) // u - v is critical there's no path for v to reach back u or previous u
        else low[u] = Math.min(low[u], dfn[v])
      }
    }
  }
}
```
#### Convert Sorted Array to Binary Search Tree
> As description 108
```javascript
const sortedArray = function(price, special, needs) {
  if (nums.length == 0) return null
    return helper(nums, 0, nums.length - 1)
    
    function helper(nums, lo, hi) {
      if (lo > hi) return null
      let mid = (lo + hi) >> 1
      let node = new TreeNode(nums[mid])
      node.left = helper(nums, lo, mid - 1)
      node.right = helper(nums, mid + 1, hi)
      return node
    }
}
```
## SORT  
### STRUCTURE CODE  
``` javascript
const quickSort = function(nums) {
  if (nums.length < 2) return nums
  let l = 0, r = nums.length - 1
  if (l < r) {
    let i = l, j = r, x = nums[l] // x pivot
    while (i < j) {
      // from right to left find the first that less than pivot
      while (i < j && nums[j] >= x) { j-- }
      if (i < j) nums[i++] = nums[j]
      // from left to right find the first that more or equal than pivot
      while (i < j && nums[i] < x) { i++ }
      if (i < j) nums[j--] = nums[i]
    }
    nums[i] = x
    quickSort(nums, l, i - 1)
    quickSort(nums, i + 1, r)
  }
  return nums
}
```
``` javascript
// more clear
const quickSort = function(nums) {
  if (nums.length < 2) return nums
  const lesser = []
  const greater = []
  const pivot = nums[0]
  for (let i = 1; i < nums.length; i++) {
    if (nums[i] < pivot) lesser.push(nums[i])
    else greater.push(nums[i])
  }
  return quickSort(lesser).concat(pivot, quickSort(greater))
}
```
```javascript
// from bottom to top
const mergeSort = function(nums) {
  if (nums.length < 2) return nums

  let step = 1
  let l = -1, r = -1
  while (step < nums.length) {
    l = 0
    r = step
    while (r + step <= nums.length) {
      merge(nums, l, l + step, r, r + step)
      l = r + step
      r = l + step
    } 
    if (r < nums.length) merge(nums, l, l + step, r, nums.length)
    step *= 2
  }
  return nums

  function merge(nums, startL, stopL, startR, stopR) {
    let rightNums = new Array(stopR - startR + 1)
    let leftNums = new Array(stopL - startL + 1)
    let k = startR
    for (let i = 0; i < rightNums.length - 1; i++) {
      rightNums[i] = nums[k]
      k++
    }
    k = startL
    for (let i = 0; i < leftNums.length - 1; i++) {
      leftNums[i] = nums[k]
      k++
    }
    // pivot value
    rightNums[rightNums.length - 1] = Infinity
    leftNums[leftNums.length - 1] = Infinity
    let m = 0
    let n = 0
    for (let k = startL; k < stopR; k++) {
      if (leftNums[m] <= rightNums[n]) {
        nums[k] = leftNums[m]
        m++
      } else {
        nums[k] = rightNums[n]
        n++
      }
    }
  }
}
```
``` javascript
// from top to bottom  
const mergeSort = function(nums) {
  if (nums.length < 2) return nums
  let mid = nums.length / 2 | 0
  let left = nums.slice(0, mid)
  let right = nums.slice(mid)
  return merge(mergeSort(left), mergeSort(right))

  function merge(left, right) {
    const res = []
    while (left.length && right.length) {
      if (left[0] <= right[0]) res.push(left.shift())
      else res.push(right.shift())
    }
    while (left.length) res.push(left.shift())
    while (right.length) res.push(right.shift())
    return res
  }
}
```
```javascript
const heapSort = function(nums) {
  const len = nums.length 

  buildMaxHeap(nums)
  for (let i = nums.length - 1; i > 0; i--) {
    swap(nums, 0, i)
    len--
    heapify(nums, 0)
  }
  return nums

  function buildMaxHeap(nums) {
    for (let i = len / 2 | 0; i >= 0; i--) { heapify(nums, i) }
  }

  function heapify(nums, i) {
    let left = 2 * i + 1
    let right = 2 * i + 2
    let largest = i
    if (left < len && nums[left] > nums[largest]) largest = left
    if (right < len && nums[right] > nums[largest]) largest = right
    if (largest !== i) {
      swap(nums, i, largest)
      heapify(nums, largest)
    }
  }

  function swap(nums, i, j) {
    let tmp = nums[i]
    nums[i] = nums[j]
    nums[j] = tmp
  }

}
```
``` javascript
const bubbleSort = function(nums) {
  const N = nums.length
  for (let i = N - 1; i > 0; i--) {
    for (let j = 0; j < i; j++) {
      // find the max to tail pos
      if (nums[j] > nums[j + 1]) swap(j, j + 1)
    }
  }
  return nums
}
```
``` javascript
const selectSort = function(nums) {
  let min = -1
  let N = nums.length
  for (let i = 0; i < N - 1; i++) {
    min = i
    for (let j = i + 1; j < N; j++) {
      // find the min to head pos
      if (nums[j] < nums[min]) min = j
    }
    swap(i, min)
  }
  return nums
}
```
``` javascript
const insertSort = function(nums) {
  const N = nums.length
  let pre = -1
  let cur = 0
  for (let i = 1; i < N; i++) {
    cur = nums[i]
    pre = i - 1
    while (pre >= 0 && nums[pre] > cur) {
      nums[pre + 1] = nums[pre]
      pre--
    }
    nums[pre + 1] = cur
  }
  return nums
}
```
``` javascript
const shellSort = function(nums) {
  const N = nums.length
  let gap = 1
  while (gap < N / 3) {
    gap = gap * 3 + 1
  }
  for (gap; gap > 0; gap = gap / 3 | 0) {
    for (let i = gap; i < N; i++) {
      let tmp = nums[i]
      for (j = i - gap; j >= 0 && nums[j] > tmp; j -= gap) {
        nums[j + gap] = nums[j]
      }
      nums[j + gap] = tmp
    }
  }
  return nums
}
```




# BASIC  
---


## NETWORK  
- **网络结构分层**  
 ***OSI 7层：*** 应用层，会话层，表示层，传输层，网络层，数据链路层，物理层  
 ***TCP/IP 4层:***  应用层，传输层，网络层，网络接口层  
 ***综合 5层：*** 应用层，传输层，网络层，数据链路层，物理层  
- **每层的作用和常用协议**  
 ***应用层：*** 通过应用进程之间的交互完成特定网络应用，该层协议定义应用进程之间的通信和交互规则
 常用协议有：**域名系统`DNS`**, **支持网络的`HTTP`**, **支持电子邮件的`SMTP`**等   
 
 - ***域名解析系统DNS：***`DNS`被设计为一个联机分布式数据库系统，并采用客户服务器方式。`DNS`使大多数名字都在本地进行解析，  
 仅少量解析需要在互联网上通信，因此`DNS`的效率很高。由于`DNS`是分布式系统，即使单个计算机出现了故障也不会妨碍到整个`DNS系统`的正常运行。  
 **解析过程：**主机向本地域名服务器的查询一般都采用递归查询，递归查询指如果主机所询问的本地域名服务器不知道被查询域名的`IP地址`，  
 那么本地域名服务器就以`DNS客户`的身份向其他根域名服务器继续发出查询请求报文。递归查询额结果是要查询的`IP地址`，或者是报错，表示无法查询到所需的`IP地址`。  
 本地域名服务器向根域名服务器查询通常采用迭代查询，迭代查询指当根域名服务器收到本地域名服务器发出的迭代查询请求报文时，要么给出所要查询的`IP地址`，  
 要么告诉它该向哪一个域名服务器进行查询。本地域名服务器也可以采用递归查询，这取决于最初的查询请求报文设置的查询方式。  
 - ***文件传送协议FTP：*** `FTP` 使用`TCP可靠的运输服务`，`FTP`使用客户服务器方式，一个`FTP服务器进程`可以同时为多个客户进程提供服务，
 在进行文件传输时，`FTP`的客户和服务器之间要建立两个并行的`TCP连接`：控制连接和数据连接，实际用于传输文件的是数据连接。  
 - ***超文本传输协议HTTP：***`HTTP`是超文本传输协议，规范了浏览器如何向万维网服务器请求万维网文档，服务器如何向浏览器发送万维网文档。  
   从层次的角度看，`HTTP`是面向事务的应用层协议，是浏览器和服务器之间的传送数据文件的重要基础。  
   **特点：**HTTP是无状态的，之所以说无状态是因为`HTTP`对事务没有记忆性。同一个客户第二次访问同一个服务器，服务器的响应结果和第一次是一样的。  
    `HTTP`的无状态简化了服务器的设计，允许服务器支持高并发的`HTTP`请求。如果要解决无状态的问题，可以使用`cookie`和`session`。  
    `Cookie`相当于服务器给浏览器的一个通行证，是一个唯一识别码，服务器发送的响应报文包含`Set-Cookie`首部字段，客户端得到响应报文后把`Cookie`内容保存到浏览器中。客户端之后对同一个服务器发送请求时，会从浏览器中取出`Cookie信息`并通过`Cookie请求首部字段`发送给服务器，
    服务器就可以识别是否是同一个客户。`Session`是服务器的会话技术，是存储在服务器的  
   **区别：**   
    ①`Cookie`只能存储`ASCII码字符串`，而`Session`则可以存储任何类型的数据，因此在考虑数据复杂性时首选`Session`  
    ②`Cookie`存储在浏览器中，容易被恶意查看。如果非要将一些隐私数据存在`Cookie`中，可以将`Cookie值`进行加密，然后在服务器进行解密  
    ③对于大型网站，如果用户所有的信息都存储在`Session`中，那么开销是非常大的，因此不建议将所有的用户信息都存储到`Session`中  

   **结构：**`HTTP报文`分为`HTTP请求报文`和`响应报文`，`请求报文`由请求行（请求方法，请求资源的URL和HTTP的版本）、首部行和实体（通常不用）组成
 `响应报文`由状态行（状态码，短语和HTTP版本）、首部行和实体（有些不用）组成  
 - **GET 和 POST**  
    *GET：*主要用于获取资源，用于访问被URI同意资源标识符识别的资源  
    *POST：*主要用于传递信息给服务器  
    *参数：*GET和POST的请求都能使用额外的参数，但是 GET 的参数是以查询字符串出现在 URL 中，而POST的参数存储在实体主体中
    不能因为 POST 参数存储在实体主体中就认为它的安全性更高，因为照样可以通过一些抓包工具查看  
    *安全性：*安全的HTTP方法不会改变服务器状态，也就是说它只是可读的。GET方法是安全的，而POST却不是，因为 POST 的目的是传送实体主体内容，
    这个内容可能是用户上传的表单数据，上传成功之后，服务器可能把这个数据存储到数据库中，因此状态也就发生了改变  
    *发送数据：*XMLHttpRequest是一个 API，在Ajax中大量使用。它为客户端提供了在客户端和服务器之间传输数据的功能，
    它提供了一个通过URL 来获取数据的简单方式，并且不会使整个页面刷新。这使得网页只更新一部分页面而不会打扰到用户。
    使用XMLHttpRequest时，GET请求发送一个TCP数据包，浏览器同时发送HTTP header和data，服务器响应状态码200。POST每次发送两个TCP数据包，
    浏览器先发送HTTP header，服务器收到后返回100（continue），浏览器再继续发送data，服务器响应200  
    PUT 上传文件 DELETE 删除文件 OPTIONS 查看当前URL支持的HTTP方法 HEAD 获取首部  
    RESTFUL  
 - **输入一个网址流程**  
 ①先检查输入的URL是否合法，然后查询浏览器的缓存，如果有则直接显示  
 ②通过DNS域名解析服务解析IP地址，先从浏览器缓存查询、然后是操作系统和hosts文件的缓存，如果没有查询本地服务器的缓存  
 ③通过TCP的三次握手机制建立连接，建立连接后向服务器发送HTTP请求，请求数据包  
 ④服务器收到浏览器的请求后，进行处理并响应  
 ⑤浏览器收到服务器数据后，如果可以就存入缓存  
 ⑥浏览器发送请求内嵌在HTML中的资源，例如css、js、图片和视频等，如果是未知类型会弹出对话框  
 ⑦浏览器渲染页面并呈现给用户  
 - **HTTP2.0和HTTP1.1的区别，HTTP2.0的原理**  
 `HTTP1.0`使用的是非持续连接，每次请求文档就有2倍的RTT开销，另外客户和服务器每一次建立新的`TCP`连接都要分配缓存和变量，  
 这种非持续连接会给服务器造成很大的压力  
 `HTTP1.1`使用的是持续连接，服务器会在发送响应后在一段时间内继续保持这条连接，  
 使同一个浏览器和服务器可以继续在这条连接上传输后续的`HTTP请求和响应报文`。`HTTP1.1`的持续连接有两种工作方式，非流水线和流水线方式  
 非流水线方式就是客户在收到前一个响应后才能发送下一个请求，流水线方式是客户收到响应前就能连着发送新的请求  
 `HTTP2.0` 特点是在不改动HTTP语义、方法、状态码、URI及首部字段的情况下，大幅度提高了web性能，基于`SPDY协议`，  
 是speed的谐音，Google开发基于`TCP协议`的应用层协议，目标是优化`HTTP性能`，通过压缩，多路复用和优先级，缩短网页加载时间并提高安全性，
 核心思想是尽量减少`TCP连接数`，对`HTTP协议`的增强  
 `HTTP1.x`缺点：  
    - HTTP/1.0一次只允许在一个TCP连接上发起一个请求，HTTP/1.1使用的流水线技术也只能部分处理请求并发，仍然会存在队列头阻塞问题,  
    因此客户端在需要发起多次请求时，通常会采用建立多连接来减少延迟。
    - 单向请求，只能由客户端发起  
    - 请求报文与响应报文首部信息冗余量大  
    - 数据未压缩，导致数据的传输量大    

   `HTTP2.0`改进  
      - 所有加强性能的核心是`二进制传输`. `HTTP1.x`都是通过文本的方式传输数据，`HTTP2.0`引入新的编码机制，所有传输的数据会被分割，  
      并采用二进制格式编码，为了保证HTTP不受影响则需要在应用层`HTTP2.0`和传输层`TCP/UDP`之间增加二进制分帧层，在该层会将传输的信息分为更小  
      的消息和帧，并采用二进制格式编码，其中`HTTP1.x`的首部信息会被封装到`Header帧`，而`RequestBody`则封装到`Data帧`  
      - `HTTP2.0`中`帧（frame）`：最小数据单位，每个帧会标识其属于哪个流和`流（stream`）：多个帧组成的数据流很重要。多路复用，即在一个`TCP连接`中  
      存在多个`流`，即可以同时发送多个请求，对端可以通过帧中的标识知道该帧属于哪个请求。在客户端这些帧乱序发送，到对端再根据每个帧首部的流标识副重新  
      组装。借此，可以避免`HTTP1.x`的队头阻塞问题，提高传输性能  
      - `Header压缩`，在`HTTP1.x中`，`Header`以文本形式传输，若其中有`Cookie`，每次开销很大， `Http2.0`中，使用了`HPACK（头部压缩算法`对`header`进行编码，  
      减少了其大小。并在两端维护索引表，用于记录出现过的Header，方便后续查找使用  
 `QUIC`是Google基于`UDP`实现的同为传输层的协议，目标是希望替代`TCP`。该协议支持多路复用，且实现了自己的加密协议，也支持重传和纠错机制（丢一个包用纠错，  
 多个就要重传，算不出来）  
 - **HTTPS**  
 `HTTP隐患：`使用明文通信，内容可能被监听，不验证对方身份，可能会被伪装通信方身份；无法证明报文完整性，可能被篡改  
 `HTTPS`让HTTP先和SSL通信，再由SSL和TCP通信，也就是说HTTPS使用了隧道进行通信。通过使用SSL，HTTPS具有了加密（防窃听）、认证（防伪装）、完整性保护（防篡改）  
 `HTTP`端口80， `HTTPS`端口443  
    - **流程**  
    加密算法主要有`对称加密`和`非对称加密`  
    对称加密的运算速度快，但安全性不高。非对称密钥加密，加密和解密使用不同的密钥。公开密钥所有人都可以获得，
    通信发送方获得接收方的公开密钥之后，就可以使用公开密钥进行加密，接收方收到通信内容后使用私有密钥解密。  
    非对称密钥除了用来加密还可以用来进行签名。因为私有密钥无法被其他人获取，因此通信发送方使用其私有密钥进行签名，通信接收方使用发送方的公开密钥对签名进行解密，
    就能判断这个签名是否正确。非对称加密的运算速度慢，但是更安全。  
    `HTTPS`采用混合的加密机制，使用`非对称密钥加密`用于*传输对称密钥来保证传输过程的安全性*，之后使用`对称密钥加密`进行*通信来保证通信过程的效率*。  
    浏览器和服务器建立`TCP连接`后，会发送一个证书请求，其中包含了自己可以实现的算法列表和一些必要信息，用于商议双方使用的加密算法。  
    服务器收到请求后会选择加密算法，然后返回证书，包含了服务器的信息，域名、申请证书的公司、加密的公钥以及加密的算法等。  
    浏览器收到之后，检查签发该证书的机构是否正确，该机构的公钥签名是否有效，如果有效就生成对称密钥，并利用公钥对其加密，然后发送给服务器。  
    服务器收到密钥后，利用自己的私钥解密。之后浏览器和服务器就可以基于对称加密对数据进行加密和通信。  

***传输层：*** 负责向两台主机进程之间的通信提供通用的数据传输服务  
 常用的协议有：  
  - **传输控制协议`TCP`**，它提供面向连接的、可靠的数据传输服务，传输单位是`报文段(segment)`  
  - **用户数据报协议`UDP`**，它提供无连接的传输服务，传输单位是`用户数据报`  
  
***网络层：*** 负责为分组交换网络上不同的主机提供通信服务，在发送数据时网络层吧运输层产生的报文段或用户数据报封装成分组或
 包进行传送。另一个任务是选择合适的路由，使源主机传输层传过来的分组能过通过网络中的路由器找到目的主机  
 常用的协议有：  
 - **网际协议`IP`**,用来使互联起来的计算机网络能够相互通信，（网际层由来）`IPV4`, `IPV6`    
 ![avatar](./pic/ipMessage.png)  
 [Details](https://github.com/CyC2018/CS-Notes/blob/master/notes/%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%BD%91%E7%BB%9C%20-%20%E7%BD%91%E7%BB%9C%E5%B1%82.md)
 - **地址解析协议`ARP`**, `IP`使用`ARP协议`，其作用是通过ARP高速缓存存储本地局域网的各主机和路由器的`IP地址`到`MAC地址`的映射表，
 以从网络层的IP地址解析出在数据链路层使用的MAC地址。`RARP`逆地址解析协议，使硬件地址的主机能够找出IP地址，被`DHCP取代`  
 - **网际控制报文协议`ICMP`**, `ICMP报文`作为IP数据报的数据，加上首部后组成`IP数据报`发送出去，ICMP允许主机或者路由器报告差错情况
 和提供有关异常情况的报告。ICMP有两种报文，`差错报告报文`和`询问报文`。其最重要的应用就是`ping`，来测试两台主机之间的连通性，ping使用了
 `ICMP回送请求`与`回送回答报文`    
 ![avatar](./pic/ICMP.png)  
 - **网际组管理协议`IGMP`**，是IP多播使用的协议，作用是让连接在本地局域网上的多播路由器知道本局域网上是否有某个进程参加或退出了某个多播组
- **TCP & UDP**  
    - ***用户数据报协议UDP：***  
      - UDP只在IP的数据报服务上增加了很少一点功能，就是复用和分用以及差错监测  
      其特点主要是：  
      - UDP是无连接的，发送数据前不需要建立连接；
      - UDP使用最大努力交付，不保证数据传输的可靠性；  
      - UDP是面向报文的，发送方UDP对应用程序交下来的报文在添加首部后就向下交付IP层；  
      - UDP没有拥塞控制；UDP支持一对一，一对多，多对一和多对多的交互通信；UDP首部开销小，只有8个字节，TCP需要20个字节  
      ![avatar](./pic/udp.png)
    - ***传输控制协议TCP：***  
      其主要特点是：  
      - TCP是面向连接的运输层协议，即TCP在进行数据通信前需要建立连接，主要是通过三次握手机制实现，在进行数据通信后，需要断开连接，通过四次挥手机制实现  
      - 每条TCP连接只能有两个端点  
      - TCP提供可靠的交付服务，通过TCP发送的数据无差错，不丢失，不重复  
      - TCP是全双工通信，在发送端和接收端没有缓存，发送发将数据发送到缓存后，接收方将数据放入缓存，上层应用程序会在合适时机获取数据  
      - TCP是面向字节流的，所谓流就是流入进程或者进程重流出的字节序列。虽然应用进程和TCP交互是一次一个数据块，但是TCP会将数据块看成  
      一连串无结构的字节流，不能保证发送的数据块和接收的数据块大小一致，但是字节流是完全一样的  
      ![avatar](./pic/tcp.png)
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
    TCP是全双工通信，任何一方都可以发起建立连接的请求，假设A是客户端，B是服务器。  
    初始时A和B均处于CLOSED状态，B会创建传输进程控制块TCB，然后处于LISTEND状态，监听端口是否收到了TCP请求以便及时响应。  
    当A要发生数据时，就向B发送一个连接请求报文，TCP规定连接请求报文的SYN=1，ACK=0，SYN表示synchronization，ACK表示acknowledgement，SYN不可以携带数据，  
    但要消耗一个序号，此时A发送的序号seq假设为x。发送完之后，A就进入了SYN-SENT同步已发送状态。  
    当B收到了A的连接请求报文后，如果B同意建立连接，会发送给A一个确认连接请求报文，其中SYN=1，ACK=1，ack=x+1，seq=y，ack的值为A发送的序号加1，ACK可以携带数据，  
    如果不携带的话，则不消耗序号。发送完之后，B进入SYN-RCVD同步已接收状态。  
    当A收到了B的确认连接请求报文后，还要对该确认再进行一次确认，报文的ACK=1，ack=y+1，seq=x+1，发送之后A处于established状态，当B接收到该报文后也进入established状态。  
    *之所以要进行三次握手*，是因为第二次握手时A知道了自己的发送和接收是没有问题的，而第三次握手时B才能知道自己的发送和接收也都是没有问题的。  
    同时三次握手防止了已失效的连接请求问题，假设这样一种正常情况，A发送的第一个连接请求报文丢失了，之后超时重传，建立了连接，通信之后释放了连接。  
    但假设A第一个发送的连接请求报文并没有丢失，而是在网络中某结点停滞了，之后又到达了B。如果是两次握手，此时B会以为是A请求建立连接，  
    同意之后并不会收到任何数据，因为A已经关闭了，此时B的资源就会被白白浪费。  
    *四次挥手*，当A已经没有要发送的数据了，决定释放连接，就会发送一个终止连接报文，其中FIN=1，seq=u，u的值为之前A发送的最后一个序号+1。此时A进入FIN-WAIT-1状态。  
    B收到该报文后，发送给A一个确认报文，ACK=1，ack=u+1，seq=v，v的值为B之前发送的最后一个序号+1。此时A进入了FIN-WAIT-2状态，但B进入了CLOSE-WAIT状态，  
    但连接并未完全释放，B会通知高层的应用层结束A到B这一方向的连接，此时TCP处于半关闭状态。  
    当B发送完数据后，准备释放连接时就向A发送连接终止报文，FIN=1，同时还要重发ACK=1，ack=u+1，seq=w（在半关闭状态B可能又发送了一些数据）。此时B进入LAST-ACK状态。  
    A收到连接终止报文后还要再进行一次确认，确认报文中ACK=1，ack=w+1，seq=u+1。发送完之后进入TIME-WAIT状态，等待2MSL之后进入CLOSED状态，B收到该确认后也进入CLOSED状态。
    MSL是最大报文段寿命，之所以要等待2MSL是为了保证A发送的最后一个ACK报文能被B接收，如果A发送的确认报文丢失，B没有收到就会超时重传之前的FIN+ACK报文，  
    而如果A在发送了确认报文之后就立即释放连接就无法收到B超时重传的报文，因而也不会再一次发送确认报文段，B就无法正常进入CLOSED状态。  
    第二点原因是2MSL时间之后，本连接中的所有报文就都会从网络中消失，防止出现三次握手中的已失效的请求报文问题，影响下一次的TCP连接。  
    之所以不是三次挥手是因为服务器TCP是全双工的，当A发送完数据之后可能B还没有发送完，当B发送完所有的数据之后才会关闭B到A方向的连接。  
    除此之外，TCP还设有一个保活计时器，用于解决服务器故障的问题，服务器每收到一次客户的数据就重新设置保活计时器，时间为2小时。  
    如果2小时内没有收到就间隔75秒发送一次探测报文，连续10次都没有响应后就关闭连接。

    - **网络拥塞控制四种算法，慢启动，拥塞避免，快重传，快恢复**  
    *慢启动*，就是基于窗口的拥塞控制，发送端设有一个拥塞窗口，拥塞窗口cwnd取决于网络的拥塞程度，发送窗口就等于拥塞窗口，初始时为了防止注入过多的数据引起网络拥塞，  
    所以将拥塞窗口值设为1，然后逐渐增大拥塞窗口，逐渐增大发送窗口，每经过一次传输轮次，拥塞窗口就加倍。结束方式一：如果存在一个由超时指示的丢包事件（即拥塞）则TCP发送方将cwnd设为1，重新开始慢启动，还将ssthresh慢启动阈值设置cwnd/2，第二种方式：当cwnd到达ssthresh时，结束慢启动，进入拥塞避免模式。最后一种方式是检测到3个duplicate ACK，  
    这时TCP执行快速重传并进入快速恢复状态。  
    *拥塞避免*就是每经过一个往返时间RRT将拥塞窗口的值增加1，而不是像慢启动那样加倍地增大拥塞窗口。慢启动不是指窗口增大的速度慢，而是在TCP开始发生报文时先设置拥塞窗口为1，  
    使发送方开始只发送一个报文段，相比一下将许多报文注入到网络慢。但是有时候个报文段丢失，而网络中并没有出现拥塞，错误地导致慢启动，降低了传输效率。这时应该使用快重传来让发送方尽早知道出现了个别分组的丢失，何时结束呢，当出现超时时，TCP的拥塞避免算法行为相同，  
    *快重传*要求接收端不要等待自己发送数据时再捎带确认，而是要立即发送确认。即使收到了乱序的报文段后也要立即发出对已收到报文段的重复确认。  
    当发送端连续收到三个重复的确认后就知道出现了报文段丢失的情况，就会立即重传，快重传可以使整个网络的吞吐量提升约20%。  
    ![avatar](./pic/fast.png)
    当发送方知道了只是丢失了个别报文段使，不会使用慢开始，而是使用*快恢复*来设置阻塞窗口的值，并开始执行拥塞避免算法。    
    *快恢复*Tahoe版本，不管发生超时指示还是3个duplicate ack，都无条件将cwnd减至1，然后慢启动。而Reno则综合了快速恢复。    
    ![avatar](./pic/tahoe_reno.png)  
    - **一些有趣的特点**  

    - **http连接中状态码有哪些，如果出现某些错误的状态码，分析出是什么情况吗**  
    [Detail](https://github.com/CyC2018/CS-Notes/blob/master/notes/HTTP.md#%E4%B8%80-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5)

## Socket  
[Detail](https://github.com/CyC2018/CS-Notes/blob/master/notes/Socket.md)  
 - **I/O模型**  
 一个输入操作通常包括两个阶段：*等待数据准备好*和*从内核向进程复制数据*   
 对于一个套接字上的输入操作，第一步通常涉及等待数据从网络中到达。当所等待数据到达时，它被复制到内核中的某个缓冲区。第二步就是把数据从内核缓冲区复制到应用进程缓冲区。  
 Unix有五种I/O模型：  
  - 阻塞式I/O  
  - 非阻塞式I/O  
  - I/O复用（select和poll）  
  - 信号驱动式I/O（SIGIO）  
  - 异步I/O（AIO）  

 - **阻塞式I/O**  
 应用进程被阻塞，直到数据从内核缓冲区复制到应用进程缓冲区中才返回  
 在阻塞的过程中，其它应用进程还可以执行，因此阻塞不意味着整个操作系统都被阻塞。因为其它应用进程还可以执行，所以不消耗 CPU 时间，这种模型的 CPU 利用率会比较高  
 - **非阻塞式I/O**  
 应用进程执行系统调用之后，内核返回一个错误码。应用进程可以继续执行，但是需要不断的执行系统调用来获知 I/O 是否完成，这种方式称为轮询（polling）  
 由于 CPU 要处理更多的系统调用，因此这种模型的 CPU 利用率比较低  
 - **I/O复用**  
 使用 select 或者 poll 等待数据，并且可以等待多个套接字中的任何一个变为可读。这一过程会被阻塞，当某一个套接字可读时返回，之后再使用 recvfrom 把数据从内核复制到进程中  
 它可以让单个进程具有处理多个 I/O 事件的能力。又被称为 Event Driven I/O，即事件驱动 I/O  
 如果一个 Web 服务器没有 I/O 复用，那么每一个 Socket 连接都需要创建一个线程去处理。如果同时有几万个连接，那么就需要创建相同数量的线程。相比于多进程和多线程技术，I/O 复用不需要进程线程创建和切换的开销，系统开销更小  
 - **信号驱动I/O**  
 应用进程使用 sigaction 系统调用，内核立即返回，应用进程可以继续执行，也就是说等待数据阶段应用进程是非阻塞的。内核在数据到达时向应用进程发送 SIGIO 信号，应用进程收到之后在信号处理程序中调用 recvfrom 将数据从内核复制到应用进程中  
 相比于非阻塞式 I/O 的轮询方式，信号驱动 I/O 的 CPU 利用率更高  
 - **异步I/O**  
 应用进程执行 aio_read 系统调用会立即返回，应用进程可以继续执行，不会被阻塞，内核会在所有操作完成之后向应用进程发送信号。异步 I/O 与信号驱动 I/O 的区别在于，异步 I/O 的信号是通知应用进程 I/O 完成，而信号驱动 I/O 的信号是通知应用进程可以开始 I/O  

 - **I/O模型比较**  
  - 同步I/O：将数据从内核缓冲区复制到应用进程缓冲区的阶段（第二阶段），应用进程会阻塞  
  - 异步I/O：第二阶段应用进程不会阻塞  
  同步 I/O 包括阻塞式 I/O、非阻塞式 I/O、I/O 复用和信号驱动 I/O ，它们的主要区别在第一个阶段  
  非阻塞式 I/O 、信号驱动 I/O 和异步 I/O 在第一阶段不会阻塞  
 ![avatar](./pic/5io.png)  

 - **I/O复用**  
 select/poll/epoll 都是 I/O 多路复用的具体实现，select 出现的最早，之后是 poll，再是 epoll  
 ```clike
 int select(int n, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout);
 ```
 select 允许应用程序监视一组文件描述符，等待一个或者多个描述符成为就绪状态，从而完成 I/O 操作。

  - fd_set 使用数组实现，数组大小使用 FD_SETSIZE 定义，所以只能监听少于 FD_SETSIZE 数量的描述符。有三种类型的描述符类型：readset、writeset、exceptset，分别对应读、写、异常条件的描述符集合  

  - timeout 为超时参数，调用 select 会一直阻塞直到有描述符的事件到达或者等待的时间超过 timeout  

  - 成功调用返回结果大于 0，出错返回结果为 -1，超时返回结果为 0  

 ```clike
 int poll(struct pollfd *fds, unsigned int nfds, int timeout);
 ```
 poll 的功能与 select 类似，也是等待一组描述符中的一个成为就绪状态  
 poll 中的描述符是 pollfd 类型的数组，pollfd 的定义如下  
 ```clike
struct pollfd {
               int   fd;         /* file descriptor */
               short events;     /* requested events */
               short revents;    /* returned events */
           };
 ```

  **比较**  
  1. 功能
    select 和 poll 的功能基本相同，不过在一些实现细节上有所不同。

    - select 会修改描述符，而 poll 不会
    - select 的描述符类型使用数组实现，FD_SETSIZE 大小默认为 1024，因此默认只能监听少于 1024 个描述符。如果要监听更多描述符的话，需要修改 FD_SETSIZE 之后重新编译；而 poll 没有描述符数量的限制  
    - poll 提供了更多的事件类型，并且对描述符的重复利用上比 select 高
    - 如果一个线程对某个描述符调用了 select 或者 poll，另一个线程关闭了该描述符，会导致调用结果不确定   
  2. 速度
    select 和 poll 速度都比较慢，每次调用都需要将全部描述符从应用进程缓冲区复制到内核缓冲区。   

  3. 可移植性
    几乎所有的系统都支持 select，但是只有比较新的系统支持 poll  

 ```clike
int epoll_create(int size);
int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event)；
int epoll_wait(int epfd, struct epoll_event * events, int maxevents, int timeout);
 ```
 epoll_ctl() 用于向内核注册新的描述符或者是改变某个文件描述符的状态。已注册的描述符在内核中会被维护在一棵红黑树上，通过回调函数内核会将 I/O 准备好的描述符加入到一个链表中管理，进程调用 epoll_wait() 便可以得到事件完成的描述符   

 从上面的描述可以看出，epoll 只需要将描述符从进程缓冲区向内核缓冲区拷贝一次，并且进程不需要通过轮询来获得事件完成的描述符  
 epoll 仅适用于 Linux OS  
 epoll 比 select 和 poll 更加灵活而且没有描述符数量限制  
 epoll 对多线程编程更有友好，一个线程调用了 epoll_wait() 另一个线程关闭了同一个描述符也不会产生像 select 和 poll 的不确定情况   

## OS  
- **基本特征**  
  - **并发**指宏观上在一段时间内能同时运行多个程序，而并行则指同一时刻能运行多个指令。
  并行需要硬件支持，如多流水线、多核处理器或者分布式计算系统。
  操作系统通过引入进程和线程，使得程序能够并发运行  
  - **共享**指系统中的资源可以被多个并发进程共同使用。
  有两种共享方式：*互斥共享*和*同时共享*。
  互斥共享的资源称为临界资源，例如打印机等，在同一时刻只允许一个进程访问，需要用同步机制来实现互斥访问  
  - **虚拟**把一个物理实体转换为多个逻辑实体。
  主要有两种虚拟技术：*时（时间）分复用技术*和*空（空间）分复用技术*。
  多个进程能在同一个处理器上并发执行使用了时分复用技术，让每个进程轮流占用处理器，每次只执行一小个时间片并快速切换。
  虚拟内存使用了*空分复用技术*，它将物理内存抽象为地址空间，每个进程都有各自的地址空间。地址空间的页被映射到物理内存，地址空间的页并不需要全部在物理内存中，当使用到一个没有在物理内存的页时，执行页面置换算法，将该页置换到内存中  
  - **异步**指进程不是一次性执行完毕，而是走走停停，以不可知的速度向前推进  

- **基本功能**  
  - **进程管理** 有进程控制，进程同步，进程通信，死锁处理，处理机调度等  
  - **内存管理** 有内存分配，地址映射，内存保护与共享，虚拟内存等  
  - **文件管理** 有文件存储空间的管理，目录管理，文件读写管理和保护等  
  - **设备管理** 完成用户的I/O请求，方便哟关乎使用各种设备，并提高设备的利用率，主要有缓冲管理，设备分配，设备处理，虚拟设备等  

- **系统调用**  
  如果一个进程在用户态需要使用内核态的功能，就进行系统调用从而陷入内核，由操作系统代为完成  
  ![avatar](./pic/call.png)
  Linux系统调用主要有  
  | Tasks | Commands |
  | ----- | -------- |
  | 进程控制 | fork(); exit(); wait(); |
  | 进程通信 | pipe(); shmget(); mmap(); |
  | 文件操作 | open(); read(); write(); |
  | 设备操作 | ioctl(); read(); write(); |
  | 信息维护 | getpid(); alarm(); sleep(); |
  | 安全 | chmod(); umask(); chown(); |

- **大内核与微内核**  
  *大内核*是将操作系统功能作为一个紧密结合的整体放到内，由于各模块共享信息，因此有很高的性能  
  *微内核*指由于操作系统不断复杂，因此将一部分操作系统功能移出内核，从而降低内核的复杂性。移出的部分根据分层的原则划分成若干服务，相互独立。在微内核结构下，操作系统被划分成小的、定义良好的模块，只有微内核这一个模块运行在内核态，其余模块运行在用户态。
  因为需要频繁地在用户态和核心态之间进行切换，所以会有一定的性能损失。

- **中断分类**  
  - **外中断** 由 CPU 执行指令以外的事件引起，如 I/O 完成中断，表示设备输入/输出处理已经完成，处理器能够发送下一个输入/输出请求。此外还有时钟中断、控制台中断等  
  - **异常**由 CPU 执行指令的内部事件引起，如非法操作码、地址越界、算术溢出等  
  - **陷入** 在用户程序中使用系统调用  

- **用户态和内核态**
  当一个任务（进程）执行系统调用而陷入内核代码中执行时，我们就称进程处于内核运行态（或简称为内核态）。此时处理器处于特权级最高的（0级）内核代码中执行。
  当进程处于内核态时，执行的内核代码会使用当前进程的内核栈。每个进程都有自己的内核栈。当进程在执行用户自己的代码时，则称其处于用户运行态（用户态）。即此时处理器在特权级最低的（3级）用户代码中运行  
  - **用户态进入内核态的方式，如何进入内核态**  
  1. 系统调用  
  这是用户态进程主动要求切换到内核态的一种方式，用户态进程通过系统调用申请使用操作系统提供的服务程序完成工作，比如fork()实际上就是执行了一个创建新进程的系统调用。而系统调用的机制其核心还是使用了操作系统为用户特别开放的一个中断来实现，例如Linux的int 80h中断  
  2. 异常  
  当CPU在执行运行在用户态下的程序时，发生了某些事先不可知的异常，这时会触发由当前运行进程切换到处理此异常的内核相关程序中，也就转到了内核态，比如缺页异常  
  3. 外围设备的中断  
  当外围设备完成用户请求的操作后，会向CPU发出相应的中断信号，这时CPU会暂停执行下一条即将要执行的指令转而去执行与中断信号对应的处理程序，如果先前执行的指令是用户态下的程序，
  那么这个转换的过程自然也就发生了由用户态到内核态的切换。比如硬盘读写操作完成，系统会切换到硬盘读写的中断处理程序中执行后续操作等。  
  - 这3种方式是系统在运行时由用户态转到内核态的最主要方式，其中系统调用可以认为是用户进程主动发起的，异常和外围设备中断则是被动的。  

- **僵尸进程**  
  - **定义：**完成了生命周期但却依然留在进程表中的进程，我们称之为 “僵尸进程”  
  - **产生：**当你运行一个程序时，它会产生一个父进程以及很多子进程。 所有这些子进程都会消耗内核分配给它们的内存和 CPU 资源。
  这些子进程完成执行后会发送一个 Exit 信号然后死掉。这个 Exit 信号需要被父进程所读取。父进程需要随后调用 wait 命令来读取子进程的退出状态，并将子进程从进程表中移除。
  若父进程正确第读取了子进程的 Exit 信号，则子进程会从进程表中删掉。但若父进程未能读取到子进程的 Exit 信号，则这个子进程虽然完成执行处于死亡的状态，但也不会从进程表中删掉  
  - **处理方式：**`ps aux | grep Z`  `kill -s SIGCHLD pid`将这里的 pid 替换成父进程的进程 id，这样父进程就会删除所有以及完成并死掉的子进程了。确保删除子僵尸的唯一方法就是杀掉它们的父进程  

- **进程与线程**  
  - **进程**  
    进程是资源分配的基本单位。一个进程由三部分组成：程序、数据及进程控制块(PCB)
    进程控制块 (Process Control Block, PCB) 描述进程的基本信息和运行状态，是进程存在的程序唯一标识,所谓的创建进程和撤销进程，都是指对 PCB 的操作  
  - **线程**  
    线程是独立调度的基本单位。
    一个进程中可以有多个线程，它们共享进程资源。
    QQ 和浏览器是两个进程，浏览器进程里面有很多线程，例如 HTTP 请求线程、事件响应线程、渲染线程等等，线程的并发执行使得在浏览器中点击一个新链接从而发起 HTTP 请求时，浏览器还可以响应用户的其它事件  
  - **区别**  
    - 拥有资源  
    进程是资源分配的基本单位，但是线程不拥有资源，线程可以访问隶属进程的资源  
    - 调度  
    线程是独立调度的基本单位，在同一进程中，线程的切换不会引起进程切换，从一个进程中的线程切换到另一个进程中的线程时，会引起进程切换  
    - 系统开销  
    由于创建或撤销进程时，系统都要为之分配或回收资源，如内存空间、I/O 设备等，所付出的开销远大于创建或撤销线程时的开销。类似地，在进行进程切换时，涉及当前执行进程 CPU 环境的保存及新调度进程 CPU 环境的设置，而线程切换时只需保存和设置少量寄存器内容，开销很小  
    - 通信方面  
    线程间可以通过直接读写同一进程中的数据进行通信，但是进程通信需要借助 IPC  

- **进程状态的切换**  
  ![avatar](./pic/process.png)  
  - **就绪状态（ready）：**等待被调度  
  - **运行状态（running）**  
  - **阻塞状态（waiting）：**等待资源  
  - 只有就`绪态和运行态`可以相互转换，其它的都是单向转换。就绪状态的进程通过调度算法从而获得 CPU 时间，转为运行状态；而运行状态的进程，在分配给它的 CPU 时间片用完之后就会转为就绪状态，等待下一次调度  
  - 阻塞状态是缺少需要的资源从而由运行状态转换而来，但是该资源*不包括* CPU 时间，缺少 CPU 时间会从运行态转换为就绪态  

- **Linux进程五种状态**  
  1. 运行(正在运行或在运行队列中等待)  
  2. 中断(休眠中, 受阻, 在等待某个条件的形成或接受到信号)  
  3. 不可中断(收到信号不唤醒和不可运行, 进程必须等待直到有中断发生)  
  4. 僵死(进程已终止, 但进程描述符存在, 直到父进程调用wait4()系统调用后释放)  
  5. 停止(进程收到SIGSTOP, SIGSTP, SIGTIN, SIGTOU信号后停止运行运行)  

- **进程通信方式**  
  - 管道( pipe )：管道是一种半双工的通信方式，数据只能单向流动，而且只能在具有亲缘关系的进程间使用。进程的亲缘关系通常是指父子或兄弟进程关系,通过pipe函数创建，fd[0]用于读，fd[1]用于写  
    ```clike
    #include <unistd.h>
    int pipe(int fd[2]);
    ```
  - FIFO/命名管道 (named pipe) ： 有名管道也是半双工的通信方式，但是它允许无亲缘关系进程间的通信,FIFO 常用于客户-服务器应用程序中，FIFO 用作汇聚点，在客户进程和服务器进程之间传递数据  
    ```clike
    #include <sys/stat.h>
    int mkfifo(const char *path, mode_t mode);
    int mkfifoat(int fd, const char *path, mode_t mode);
    ```
  - 消息队列( message queue ) ： 消息队列是由消息的链表，存放在内核中并由消息队列标识符标识。消息队列克服了信号传递信息少、而相比于管道，消息队列有如下有点
    - FIFO只能承载无格式字节流以及缓冲区大小受限  
    - 消息队列可以独立于读写进程存在，从而避免了 FIFO 中同步管道的打开和关闭时可能产生的困难  
    - 避免了 FIFO 的同步阻塞问题，不需要进程自己提供同步方法  
    - 读进程可以根据消息类型有选择地接收消息，而不像 FIFO 那样只能默认地接收  
  - 信号量( semophore ) ： 信号量是一个计数器，可以用来控制多个进程对共享资源的访问。它常作为一种锁机制，防止某进程正在访问共享资源时，其他进程也访问该资源。因此，主要作为进程间以及同一进程内不同线程之间的同步手段  
  - 信号(sinal) ：信号是一种比较复杂的通信方式，用于通知接收进程某些事件已经发生，要注意信号处理中调用的函数是否为信号安全  
  - 共享内存( shared memory ) ：共享内存就是映射一段能被其他进程所访问的内存，这段共享内存由一个进程创建，但多个进程都可以访问。共享内存是最快的 IPC 方式，它是针对其他进程间通信方式运行效率低而专门设计的。它往往与其他通信机制，如信号两，配合使用，来实现进程间的同步和通信  
  - 套接字( socket ) ： 套解口也是一种进程间通信机制，与其他通信机制不同的是，它可用于不同及其间的进程通信  
  **最快的方式是共享内存**：采用共享内存通信的一个显而易见的好处是效率高，因为进程可以直接读写内存，而不需要任何数据的拷贝。对于像管道和消息队列等通信方式，则需要在内核和用户空间进行四次的数据拷贝

- **进程调度算法**  
  - **批处理系统** 批处理系统没有太多的用户操作，在该系统中，调度算法目标是保证吞吐量和周转时间（从提交到终止的时间）  
    - 先来先服务 first-come first-serverd（FCFS）  
      非抢占式的调度算法，按照请求的顺序进行调度。有利于长作业，但不利于短作业，因为短作业必须一直等待前面的长作业执行完毕才能执行，而长作业又需要执行很长时间，造成了短作业等待时间过长  
    - 短作业优先 shortest job first（SJF）  
      非抢占式的调度算法，按估计运行时间最短的顺序进行调度。长作业有可能会饿死，处于一直等待短作业执行完毕的状态。因为如果一直有短作业到来，那么长作业永远得不到调度  
    - 最短剩余时间优先 shortest remaining time next（SRTN）  
      最短作业优先的抢占式版本，按剩余运行时间的顺序进行调度。 当一个新的作业到达时，其整个运行时间与当前进程的剩余时间作比较。如果新的进程需要的时间更少，则挂起当前进程，运行新的进程。否则新的进程等待  
  - **交互式系统** 交互式系统有大量的用户交互操作，在该系统中调度算法的目标是快速地进行响应  
    - 时间片轮转  
      将所有就绪进程按 FCFS 的原则排成一个队列，每次调度时，把 CPU 时间分配给队首进程，该进程可以执行一个时间片。当时间片用完时，由计时器发出时钟中断，调度程序便停止该进程的执行，并将它送往就绪队列的末尾，同时继续把 CPU 时间分配给队首的进程  
      时间片轮转算法的效率和时间片的大小有很大关系：1. 因为进程切换都要保存进程的信息并且载入新进程的信息，如果时间片太小，会导致进程切换得太频繁，在进程切换上就会花过多时间 2.而如果时间片过长，那么实时性就不能得到保证  
    - 优先级调度  
      为每个进程分配一个优先级，按优先级进行调度。为了防止低优先级的进程永远等不到调度，可以随着时间的推移增加等待进程的优先级  
    - 多级反馈队列  
      一个进程需要执行 100 个时间片，如果采用时间片轮转调度算法，那么需要交换 100 次。多级队列是为这种需要连续执行多个时间片的进程考虑，它设置了多个队列，每个队列时间片大小都不同，例如 1,2,4,8,..。进程在第一个队列没执行完，
      就会被移到下一个队列。这种方式下，之前的进程只需要交换 7 次。每个队列优先权也不同，最上面的优先权最高。因此只有上一个队列没有进程在排队，才能调度当前队列上的进程。可以将这种调度算法看成是时间片轮转调度算法和优先级调度算法的结合  
    - 实时系统  
      实时系统要求一个请求在一个确定时间内得到响应。分为硬实时和软实时，前者必须满足绝对的截止时间，后者可以容忍一定的超时  

- **线程的状态**  
  线程通常都有五种状态，创建、就绪、运行、阻塞和死亡  
  第一是创建状态。在生成线程对象，并没有调用该对象的start方法，这是线程处于创建状态  
  第二是就绪状态。当调用了线程对象的start方法之后，该线程就进入了就绪状态，但是此时线程调度程序还没有把该线程设置为当前线程，此时处于就绪状态。在线程运行之后，从等待或者睡眠中回来之后，也会处于就绪状态  
  第三是运行状态。线程调度程序将处于就绪状态的线程设置为当前线程，此时线程就进入了运行状态，开始运行run函数当中的代码  
  第四是阻塞状态。线程正在运行的时候，被暂停，通常是为了等待某个时间的发生（比如说某项资源就绪）之后再继续运行。sleep,suspend等方法都可以导致线程阻塞  
  第五是死亡状态。如果一个线程的run方法执行结束，该线程就会死亡。对于已经死亡的线程，无法再使用start方法令其进入就绪状态  

- **进程同步**  
  - **临界区**
    对临界资源进行访问的那段代码称为临界区。为了互斥访问临界资源，每个进程在进入临界区之前，需要先进行检查  
    ```clike
    // entry section
    // critical section;
    // exit section
    ```  
  - **同步与互斥**  
    - 同步：多个进程因为合作产生的直接制约关系，使得进程有一定的先后执行关系  
    - 互斥：多个进程在同一时刻只有一个进程能进入临界区  
  - **信号量**  
    信号量（Semaphore）是一个整型变量，可以对其进行`P`和`V`操作  
    `P`:如果信号量大于 `0` ，执行 `-1` 操作；如果信号量等于 `0`，进程睡眠，等待信号量大于 `0`  
    `V`:对信号量执行 `+1` 操作，唤醒睡眠的进程让其完成 `P` 操作  
    通常执行这些操作时需要进行屏蔽中断  
    如果信号量取值只能为0/1，那么就成为了**互斥量（Mutex）**，`0`表示临界区已经上锁，`1`表示临界区解锁  
    ```clike
    typedef int semaphore;
    semaphore mutex = 1;
    void P1() {
        P(&mutex);
        // 临界区
        V(&mutex);
    }

    void P2() {
        P(&mutex);
        // 临界区
        V(&mutex);
    }
    ```
    **Example:使用信号量实现生产者-消费者问题**  
    >问题描述：使用一个缓冲区来保存物品，只有缓冲区没有满，生产者才可以放入物品；只有缓冲区不为空，消费者才可以拿走物品。因为缓冲区属于临界资源，因此需要使用一个互斥量 mutex 来控制对缓冲区的互斥访问。
    >为了同步生产者和消费者的行为，需要记录缓冲区中物品的数量。数量可以使用信号量来进行统计，这里需要使用两个信号量：empty 记录空缓冲区的数量，full 记录满缓冲区的数量。其中，empty 信号量是在生产者进程中使用，
    >当 empty 不为 0 时，生产者才可以放入物品；full 信号量是在消费者进程中使用，当 full 信号量不为 0 时，消费者才可以取走物品。
    >注意，不能先对缓冲区进行加锁，再测试信号量。也就是说，不能先执行 P(mutex) 再执行 P(empty)。如果这么做了，那么可能会出现这种情况：生产者对缓冲区加锁后，执行 P(empty) 操作，
    >发现 empty = 0，此时生产者睡眠。消费者不能进入临界区，因为生产者对缓冲区加锁了，消费者就无法执行 V(empty) 操作，empty 永远都为 0，导致生产者永远等待下，不会释放锁，消费者因此也会永远等待下去。
    ```clike
    #define N 100
    typedef int semaphore;
    semaphore mutex = 1;
    semaphore empty = N;
    semaphore full = 0;

    void producer() {
        while(TRUE) {
            int item = produce_item();
            P(&empty);
            P(&mutex);
            insert_item(item);
            V(&mutex);
            V(&full);
        }
    }

    void consumer() {
        while(TRUE) {
            P(&full);
            P(&mutex);
            int item = remove_item();
            consume_item(item);
            V(&mutex);
            V(&empty);
        }
    }
    ```
  - **管程**  
    使用信号量机制实现的生产者消费者问题需要客户端代码做很多控制，而管程把控制的代码独立出来，不仅不容易出错，也使得客户端代码调用更容易  
    管程有一个重要特性：在一个时刻只能有一个进程使用管程。进程在无法继续执行的时候不能一直占用管程，否则其它进程永远不能使用管程。
    管程引入了 **条件变量** 以及相关的操作：`wait()` 和 `signal()` 来实现同步操作。对条件变量执行 `wait()` 操作会导致调用进程阻塞，把管程让出来给另一个进程持有。`signal()` 操作用于唤醒被阻塞的进程  

- **经典同步问题**  
  - **哲学家进餐问题**  
    >五个哲学家围着一张圆桌，每个哲学家面前放着食物。哲学家的生活有两种交替活动：吃饭以及思考。当一个哲学家吃饭时，需要先拿起自己左右两边的两根筷子，并且一次只能拿起一根筷子  
    
    为防止出现死锁，设置两个条件：1. 必须同时拿起左右两根筷子 2. 只有在两个邻居都没有进餐的情况下才可以进餐  
    ```clike
    #define N 5
    #define LEFT (i + N - 1) % N // 左邻居
    #define RIGHT (i + 1) % N    // 右邻居
    #define THINKING 0
    #define HUNGRY   1
    #define EATING   2
    typedef int semaphore;
    int state[N];                // 跟踪每个哲学家的状态
    semaphore mutex = 1;         // 临界区的互斥，临界区是 state 数组，对其修改需要互斥
    semaphore s[N];              // 每个哲学家一个信号量

    void philosopher(int i) {
        while(TRUE) {
            think(i);
            take_two(i);
            eat(i);
            put_two(i);
        }
    }

    void take_two(int i) {
        P(&mutex);
        state[i] = HUNGRY;
        check(i);
        up(&mutex);
        V(&s[i]); // 只有收到通知之后才可以开始吃，否则会一直等下去
    }

    void put_two(i) {
        P(&mutex);
        state[i] = THINKING;
        check(LEFT); // 尝试通知左右邻居，自己吃完了，你们可以开始吃了
        check(RIGHT);
        V(&mutex);
    }

    void eat(int i) {
        P(&mutex);
        state[i] = EATING;
        V(&mutex);
    }

    // 检查两个邻居是否都没有用餐，如果是的话，就 V(&s[i])，使得 P(&s[i]) 能够得到通知并继续执行
    void check(i) {         
        if(state[i] == HUNGRY && state[LEFT] != EATING && state[RIGHT] !=EATING) {
            state[i] = EATING;
            V(&s[i]);
        }
    }
    ```
  - **读者-写者问题**  
    允许多个进程同时对数据进行读操作，但是不允许读和写以及写和写操作同时发生。一个整型变量 `count` 记录在对数据进行读操作的进程数量，一个互斥量 `count_mutex` 用于对 `count` 加锁，一个互斥量 `data_mutex` 用于对读写的数据加锁  
    ```clike
    typedef int semaphore;
    semaphore count_mutex = 1;
    semaphore data_mutex = 1;
    int count = 0;

    void reader() {
        while(TRUE) {
            P(&count_mutex);
            count++;
            if(count == 1) P(&data_mutex); // 第一个读者需要对数据进行加锁，防止写进程访问
            V(&count_mutex);
            read();
            P(&count_mutex);
            count--;
            if(count == 0) V(&data_mutex);
            V(&count_mutex);
        }
    }

    void writer() {
        while(TRUE) {
            P(&data_mutex);
            write();
            V(&data_mutex);
        }
    }
    ```
- **死锁**  
  - **必要条件**  
    - 互斥：每个资源要么已经分配给了一个进程，要么就是可用的  
    - 占有和等待：已经得到了某个资源的进程可以再请求新的资源  
    - 不可抢占：已经分配给一个进程的资源不能强制性地被抢占，它只能被占有它的进程显式地释放  
    - 环路等待：有两个或者两个以上的进程组成一条环路，该环路中的每个进程都在等待下一个进程所占有的资源  