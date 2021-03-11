import glob
from shutil import copy
import os

root_dir = r"C:\Users\soki\Documents\TAU\DL\proj\txt"
target_dir = r"C:\Users\soki\Documents\TAU\DL\proj\txt_files"


# print(glob.glob(root_dir + "\*\*\*\*\*\*\merged.txt", recursive=True))
print(glob.glob(root_dir + "\**\merged.txt", recursive=True))

# paths = ["\*\*\*\merged.txt", "\*\*\*\*\merged.txt", "\*\*\*\*\*\merged.txt"]
# for path in paths:
#     for filename in
for filename in glob.glob(root_dir + "\**\merged.txt", recursive=True):
    target_name = "_".join(os.path.basename(os.path.dirname(os.path.dirname(filename))).split()).lower() + ".txt"
    copy(filename, target_dir + "\\" + target_name)