def transpose(lst2):
    x = 0
    main_lst = []
    if x != len(lst2):
        for i in range(len(lst2[0])):
            lst = []
            for j in range(len(lst2)):
                lst.append(lst2[j][x])
            main_lst.append(lst)
            x += 1
        return main_lst
# time complexity of this algorithm is O(m * n) where m is the no of rows and n is the no of columns
def Multiplication(lst1, lst2):
    def lst_multiplication(l1, l2):
        a = 0
        for i in range(len(l1)):
            a += l1[i] * l2[i]
        var = a
        return var
    lst2 = transpose(lst2)
    main_lst = []
    for k in range(len(lst1)):
        ele = []
        for l in range(len(lst2)):
            ele.append(lst_multiplication(lst1[k], lst2[l]))
        main_lst.append(ele)
    return main_lst
# time complexity of this algorithm is O(m * n * p)
# w here m is the no of rows of matrix 1, n no of columns of matrix 2 and p no of rows of matrix 2
flag = [
    [1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]
]
pointer = [
    [0.2, 0.5, -0.26],
    [0.8,-0.91,-0.27],
    [-0.5, 0.26, 0.17],
    [1, -0.5, 0.87]
]
print(Multiplication(flag, pointer))
