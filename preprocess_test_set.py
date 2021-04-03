import re
from glob import glob
import os
import numpy

def my_filter(line):
    if len(line) < 30:
        return False
    if "damage" in line or "break" in line or "vacat" in line or "[" in line:
        return False
    return True

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