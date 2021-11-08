from math import log10, ceil


def bin_alex(A, i, _=2):
    N = len(A)
    left = 2 * i + 1
    right = 2 * i + 2
    index = i
    if left < N and A[left] > A[index]:
        index = left
    if right < N and A[right] > A[index]:
        index = right
    if index != i:
        # thx python
        A[i], A[index] = A[index], A[i]
        bin_alex(A, index)


def binary_HEAPIFY(array, i):
    arrlen_ = len(array)
    l, r = 2*i + 1, 2*i+2
    index = i
    if l < arrlen_ and array[l] > array[index]:
        index = l
    if r < arrlen_ and array[r] > array[index]:
        index = r
    if i != index:
        array[i], array[index] = array[index], array[i]
        binary_HEAPIFY(array, index)


def HEAPIFY(arr, i, d):
    n = len(arr)
    index = i
    for j in range(1,d+1):
        k=d*i+j
        if k < n and arr[k] > arr[index]:
            index = k
    if i != index:
        temp = arr[index]
        arr[index] = arr[i]
        arr[i] = temp
        HEAPIFY(arr,index,d)

def is_heap(A, d):
    length = len(A)
    for i in range(length):
        for d_ctr in range(1, d+1):
            if d*i + d_ctr >= length:
                return True
            if A[i] < A[d*i + d_ctr]:
                return False


def heap_height(arr, d):
    arrlen = len(arr)
    return ceil(log10(arrlen*(d-1)+1)/log10(d))


def print_heap(array, d, row=0, maxstrlen=0):
    arr = array[::-1]  # reverse to print an actual tree
    TOPPAD = "_"
    MAXSTRLEN = len(str(max(arr))) if maxstrlen == 0 else maxstrlen
    trunk = (sum([d**i for i in range(0, (heap_height(arr, d))-1)]))
    leaves = len(arr)-trunk
    for i, n in enumerate(arr):
        # top row
        if i == 0 and row:
            print(" "*(MAXSTRLEN*((row-1)*(d-1))+1), end="")
        if i < leaves:
            padding_len = (1+MAXSTRLEN-len(str(n))) if row == 0 else (row *
                                                                  (d-1)*2)*(1+MAXSTRLEN-len(str(n)))
            padding_len = padding_len if i <leaves-1 else 0
            padding = TOPPAD*padding_len
            if i%d==0 and i !=0:
                padding=" "*padding_len
            print(str(n)+padding, end="")
        else:
            print()
            print_heap(array[:trunk], d, (row+1), MAXSTRLEN)
            break


def build_heap(array, func=binary_HEAPIFY, d=2):
    start = sum([d**i for i in range(0, (heap_height(array, d))-1)])
    while start >= 0:
        if func == binary_HEAPIFY:
            func(array, start)
        else:
            func(array, start, d)
        print(start)
        start -= 1


# for binary-heapify
arr = [1, 2, 3, 4, 5, 6, 7, 80, 90]

if __name__ == "__main__":
    D = 3
    build_heap(arr, HEAPIFY, d=D)
    print(arr, heap_height(arr, 2))
    print_heap(arr, D)
