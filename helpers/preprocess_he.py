import re
with open(r"C:\Users\soki\Documents\TAU\DL\proj\txt_files\all.txt", "r", encoding="utf-8") as f:
    data = f.read()#[:10000]

result = re.sub("[^א-ת\.\s]", "", data)
result = re.sub("\.", "\n", result)
# result = re.sub("\n\n", "\n", result)
# result = result.replace("\n\n", "\n")
lines = result.split("\n")
thresh = 15
lines = [line for line in lines if len(line) > thresh]


with open(r"C:\Users\soki\Documents\TAU\DL\proj\txt_files\all2.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))