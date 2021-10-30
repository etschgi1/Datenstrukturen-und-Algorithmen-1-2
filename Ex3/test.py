import numpy as np
import random
MAX = 5
Size = 5
y = [str(x)+" -" for x in range(1, Size+1)]
c = [int(random.random()*MAX % MAX)+1 for i in range(Size)]
print(y)
print(c)


def merge_sort(words, counts, start, end):
    if start < end:
        k = int((start+end)/2)
        merge_sort(words, counts, start, k)
        merge_sort(words, counts, k+1, end)
        return merge_(words, counts, start, k, end)
    else:
        return words, counts


def merge_(words, counts, start, k, end):
    left_arr, right_arr = [], []
    for left in range(start, k+1):  # range-stop is exclusive in python
        left_arr.append((words[left], counts[left]))  # append ist O(1)
    for right in range(k+1, end+1):
        right_arr.append((words[right], counts[right]))
    left_arr.append(("", -np.inf))  # add inf to the end
    right_arr.append(("", -np.inf))
    left, right = 0, 0
    for counter in range(start, end+1):
        if left_arr[left][1] >= right_arr[right][1]:
            words[counter], counts[counter] = left_arr[left][0], left_arr[left][1]
            left += 1
        else:
            words[counter], counts[counter] = right_arr[right][0], right_arr[right][1]
            right += 1
    return words, counts


y, c = merge_sort(y, c, 0, len(y)-1)
print()
print(y)
print(c)
