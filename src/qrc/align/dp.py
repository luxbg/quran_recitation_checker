"""Edit-distance alignment with backpointers, small-buffer O(n*m) DP.

Buffers passed in here are kept small by the caller (a handful of words at a
time, trimmed after each settled word) so a plain full DP is fast enough --
no literal Ukkonen banding needed at this scale.
"""

import numpy as np

MATCH = 0
INSERT_A = 1  # a[i-1] is extra, not present in b (recited something extra / ASR noise)
DELETE_A = 2  # b[j-1] has no counterpart in a (skipped / not recited)


def edit_distance_with_backpointers(a: str, b: str) -> tuple[np.ndarray, np.ndarray]:
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    bp = np.full((n + 1, m + 1), -1, dtype=np.int8)

    for i in range(1, n + 1):
        dp[i][0] = i
        bp[i][0] = INSERT_A
    for j in range(1, m + 1):
        dp[0][j] = j
        bp[0][j] = DELETE_A

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            sub_cost = 0 if ai == b[j - 1] else 1
            diag = dp[i - 1][j - 1] + sub_cost
            up = dp[i - 1][j] + 1
            left = dp[i][j - 1] + 1

            best = diag
            op = MATCH
            if up < best:
                best, op = up, INSERT_A
            if left < best:
                best, op = left, DELETE_A

            dp[i][j] = best
            bp[i][j] = op

    return dp, bp


def traceback(bp: np.ndarray, i: int, j: int) -> list[tuple[int | None, int | None, int]]:
    path: list[tuple[int | None, int | None, int]] = []
    while i > 0 or j > 0:
        op = int(bp[i][j])
        if op == MATCH:
            path.append((i - 1, j - 1, MATCH))
            i -= 1
            j -= 1
        elif op == INSERT_A:
            path.append((i - 1, None, INSERT_A))
            i -= 1
        elif op == DELETE_A:
            path.append((None, j - 1, DELETE_A))
            j -= 1
        else:
            break
    path.reverse()
    return path
