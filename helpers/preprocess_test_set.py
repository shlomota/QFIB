import re
from glob import glob
import os
import numpy

def my_filter(line):
    # illegal_li = ["damage", "×", "x", "-", "־", "break", "vacat", "[", "]"]
    min_len = 30
    if len(line) < min_len:
        return False
    if re.search("[^א-ת \n]", line):
            return False

    return True

do_all = False

if do_all:
    bad_list = glob(r"C:\Users\soki\Documents\TAU\DL\proj\qumran\megilot\*-index*")
    # print(len(bad_list))
    good_list = glob(r"C:\Users\soki\Documents\TAU\DL\proj\qumran\megilot\*")
    # print(len(good_list))
    good_list = list(set(good_list).difference(bad_list))
    # print(len(good_list))


    sentences = []
    for filename in good_list:
        with open(filename, "r", encoding="utf8") as f:
            data = f.readlines()
        # data = [(line, os.path.basename(filename)) for line in data if my_filter(line)]
        data = [line for line in data if my_filter(line)]

        sentences += data
    # for f in glob(r"C:\Users\soki\Documents\TAU\DL\proj\qumran\megilot\*-index*"):
    #     print(f)
    print(len(sentences))
    with open(r"C:\Users\soki\PycharmProjects\QFIB\data\test_data.txt", "w", encoding="utf8") as f:
        f.writelines(sentences)

else:
    filename = r"C:\Users\soki\Documents\TAU\DL\proj\qumran\megilot\CD.txt"
    with open(filename, "r", encoding="utf8") as f:
        data = f.readlines()
    data = [line for line in data if my_filter(line)]
    print(len(data))
    with open(r"C:\Users\soki\PycharmProjects\QFIB\data\test_data_CD.txt", "w", encoding="utf8") as f:
        f.writelines(data)