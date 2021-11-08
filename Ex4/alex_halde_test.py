import math
import random
import datetime
from time import time
from matplotlib import pyplot as plt
import os
import numpy as np
FP = os.path.abspath(os.path.dirname(__file__))
try:
    os.makedirs((FP+r'\np_out'))
except FileExistsError:
    pass
print(FP)
NR_TESTS = 5
NR_RUNS = 100
MAX_N = 1000000 #Longest Lists
D_MIN = 2
D_MAX = 10

DEBUG_PRINT = 1 # 0,1,2 allowed

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# --------------------IMPLEMENT YOU SOLUTION HERE--------------------------


def n_ary_heapify(arr, i, d):
    n = len(arr)
    index = i
    for j in range(1, d+1):
        k = d*i+j
        if k < n and arr[k] > arr[index]:
            index = k
    if i != index:
        temp = arr[index]
        arr[index] = arr[i]
        arr[i] = temp
        n_ary_heapify(arr, index, d)
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------


# translated from lecture-slides
def binary_heapify(A, i):
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
        binary_heapify(A, index)


# translated from lecture-slides
def build_heap(A, d):
    for i in range(math.floor(len(A) / 2) - 1, -1, -1):
        n_ary_heapify(A, i, d)


def is_heap(A, d):
    length = len(A)
    for i in range(length):
        for d_ctr in range(1, d + 1):
            if d * i + d_ctr >= length:
                return True
            if A[i] < A[d * i + d_ctr]:
                return False


def random_list(n):
    A = []
    for i in range(n):
        n = random.randint(1, 30)
        A.append(n)
    return A


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
def test(NR_TESTS,n,d,DEBUG_PRINT):
    arr = []
    res = []
    successful_tests = 0
    print("-------- RUNNING -------- n = ",n,"D = ",d)
    time_ = []
    for it in range(NR_TESTS):
        success = False
        # randomly generate list and d
        D = d#random.randint(2, 10)
        arr = random_list(n)
        oldarr = arr
        start = time()
        build_heap(arr, D)
        time_.append(time()-start)
        if is_heap(arr, D):
            successful_tests += 1
            success = True

        if(DEBUG_PRINT>=2):
            print("build heap with list:", arr)
            print("and with d:", D)
            print("transformed list:", arr)
            print("SUCCESSFUL:", success)
            print()
        if not success:
            raise Exception(f"Test not passed!!!\nArr(in): {oldarr}\n Arr(out): {arr}")
    if(DEBUG_PRINT>0):
        print("-------- DONE --------")
        print("-------------------------------")
        print("TESTCASES PASSED:", successful_tests,
            "OUT OF", len(NR_TESTS), "\nTime: ", (time()-start), "s")
        print("-------------------------------")
        print("-------------------------")
        print("-------------------")
        print("-------------")
        print("-------")
    return (sum(time_)) #return time

def test_wrapper(Tests,NR_RUNS,Result_list = [],DEBUG_PRINT=False):
    #appends to given Result_list
    out = {}
    for D in range(D_MIN,D_MAX+1):
        for Test in Tests:
            Result_list.append((Test,test(NR_TESTS,Test,D,DEBUG_PRINT)))
        out[D] = Result_list 
    return out 


def plot_res(res):
    c = 0
    for key in res.keys():
        print(f"D = {key}")
        vals = res[key]
        l = len(vals)//(D_MAX-D_MIN+1)
        s,e = c,c+l
        x,y = [x[0] for x in vals[s:e]],[y[1] for y in vals[s:e]]
        lab = f"{key}-wertige Heaps"
        print(f"{key} - vals:\n{x}\n{y}")
        plt.plot(x,y,label=lab)
        print(vals[-1][1],(key*np.log(x[-1])/np.log(key)))
        norm = key*(np.log(x[-5])/np.log(key))/vals[-5][1]
        plt.plot(x,(key*np.log(x)/np.log(key)/norm))
        now = datetime.datetime.now()
        file = f"\{key}-{now.day}-{now.month}-{now.year}_{now.hour}-{now.minute}.npy"
        file = FP+r'\np_out'+file
        print("Saved to: ",file)
        np.save(file,np.array(vals))
        c+=l
    xlab = r'$n$ Länge des Arrays / n'
    ylab = r'$t$ Laufzeit / s'
    plt.grid()
    # plt.yscale("log")
    plt.legend()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title("Laufzeit einer d-ären Verhaldung")
    plt.show()




if __name__ == "__main__":
    # Test = np.linspace(1,int(MAX_N),10,dtype=int)
    Test = np.logspace(1,6,20,dtype=int)
    print(Test)
    print(Test)
    plot_res(test_wrapper(Test,DEBUG_PRINT)) 