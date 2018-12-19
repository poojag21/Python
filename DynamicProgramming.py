# -*- coding: utf-8 -*-

import numpy as np

# ============================== Counting Pond ================================


def count_ponds(G):
    m = len(G)
    n = len(G[0])
    count = 0
    visited = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            if G[i][j] == '#' and visited[i][j] == 0:
                count += 1
                pond_length(G, i, j, visited, count)
    return count


def pond_length(G, i, j, visited, count):
    visited[i][j] = count
    for t in range(i-1, i+2):
        for k in range(j-1, j+2):
            if t >= 0 and k >= 0 and t+1 <= len(G) \
                    and k+1 <= len(G[0]) and visited[t][k] == 0:
                if G[t][k] == '#':
                    pond_length(G, t, k, visited, count)

# ======================== Longest Ordered Subsequence ========================


def longest_ordered_subsequence(L):
    n = len(L)
    lis = [1] * n
    for i in range(1, n):
        for j in range(i):
            if L[i] > L[j]:
                lis[i] = max(lis[j]+1, lis[i])
    length = max(lis)
    return length

# =============================== Supermarket =================================


def supermarket(Items):
        n = len(Items)
        Items.sort(reverse=True)
        print(Items)
        max_deadline = 0
        for i in range(n):
            if Items[i][1] >= max_deadline:
                max_deadline = Items[i][1]
        time_slot = [0] * (max_deadline + 1)
        for i in range(n):
            k = min(max_deadline, Items[i][1])
            if k >= 1 and time_slot[k] == 0:
                time_slot[k] = Items[i][0]
        maximum_profit = sum(time_slot)
        return maximum_profit

# =============================== Unit tests ==================================


def test_suite():

    if count_ponds(["#--------##-",
                    "-###-----###",
                    "----##---##-",
                    "---------##-",
                    "---------#--",
                    "--#------#--",
                    "-#-#-----##-",
                    "#-#-#-----#-",
                    "-#-#------#-",
                    "--#-------#-"]) == 3:
        print('passed')
    else:
        print('failed')

    if longest_ordered_subsequence([1, 7, 3, 5, 9, 4, 8]) == 4:
        print('passed')
    else:
        print('failed')

    if supermarket([(50, 2), (10, 1), (20, 2), (30, 1)]) == 80:
        print('passed')
    else:
        print('failed')

    if supermarket([(20, 1), (2, 1), (10, 3), (100, 2),
                    (8, 2), (5, 20), (50, 10)]) == 185:
        print('passed')
    else:
        print('failed')

    if longest_ordered_subsequence([3, 10, 2, 1, 20]) == 3:
        print('passed')
    else:
        print('failed')

    if count_ponds(["#--------##-",
                    "-###-----###",
                    "----##------",
                    "---------##-",
                    "------------",
                    "--#------#--",
                    "-#-#-----##-",
                    "----------#-",
                    "-#-#------#-",
                    "--#-------#-"]) == 6:
        print('passed')
    else:
        print('failed')


if __name__ == '__main__':
    test_suite()
