def count_vectorizer(texts, k=1):
    y = []
    c = []
    # TODO begin
    entries = 0
    vals = {}
    for text in texts:
        counter, len_ = 0, len(text)
        while counter <= (len_-k):  # info to avoid running index over array bound
            word = ""
            for i in range(k):
                word += text[counter+i]+" "
            word = word.rstrip()
            try:
                vals[word] += 1
            except KeyError:
                vals[word] = 1
            counter += 1
        entries += counter
    y, c = list(vals.keys()), list(vals.values())
    # check if already sorted
    arrlen_ = len(y)
    if arrlen_ == entries:  # info already sorted fast return
        return y, c

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
        left_arr.append(("", -inf))  # -inf so first who gets to -inf will
        right_arr.append(("", -inf))  # always loose comparison
        left, right = 0, 0
        for counter in range(start, end+1):
            if left_arr[left][1] >= right_arr[right][1]:  # order
                words[counter], counts[counter] = left_arr[left][0], left_arr[left][1]
                left += 1
            else:
                words[counter], counts[counter] = right_arr[right][0], right_arr[right][1]
                right += 1
        return words, counts
    y, c = merge_sort(y, c, 0, arrlen_-1)  # call functions
    # TODO end
    return y, c
