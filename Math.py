def transpose(lst2, main_lst):
    x = 0
    if x != len(lst2):
        for i in range(len(lst2[0])):
            lst = []
            for j in range(len(lst2)):
                lst.append(lst2[j][x])
            main_lst.append(lst)
            x += 1
        return main_lst


def Multiplication(lst1, lst2, a_lst):
    def lst_multiplication(l1, l2):
        a = 0
        var = 0
        for i in range(len(l1)):
            a += l1[i] * l2[i]
        var = a
        return var

    x = 0
    ele = []
    for k in range(len(main_lst[0])):
        for l in range(len(lst2)):
            a_lst.append(lst_multiplication(lst1[x], lst2[l]))
        x += 1
    return a_lst