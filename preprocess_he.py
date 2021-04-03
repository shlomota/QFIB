import re
with open(r"C:\Users\soki\PycharmProjects\QFIB\data\he\wikidata.txt", "r", encoding="utf-8") as f:
    data = f.read()[:10000]

result = re.sub("[^א-ת\.\s]", "", data)
a=5


with open(r"C:\Users\soki\PycharmProjects\QFIB\data\he\wikidata2.txt", "w", encoding="utf-8") as f:
    f.write(result)