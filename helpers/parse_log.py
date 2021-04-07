import re
import matplotlib.pyplot as plt

path = r"C:\Users\soki\PycharmProjects\QFIB\logs\train_11180.out"
with open(path, "r") as f:
    data = f.readlines()

loss = []
epoch = []

for line in data[:-1]:
    nums = re.findall("[\d\.]+", line)
    loss += [float(nums[0])]
    epoch += [float(nums[-1])]

ax = plt.figure()
plt.plot(epoch, loss)

# custom_ticks = np.linspace(ymin, ymax, N, dtype=int)
# ax.set_xticks(10)
plt.locator_params(axis="x", numticks=10)
plt.show()