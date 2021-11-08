def heapify(A, i, d):
    n = len(A)
    index = i
    for j in range(1, d+1):
        k = d*i+j
        if k < n and A[k] > A[index]:
            index = k
    if i != index:
        temp = A[index]
        A[index] = A[i]
        A[i] = temp
        heapify(A, index, d)