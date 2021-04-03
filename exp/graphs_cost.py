import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['LR', 'PCA', 'Power Set', 'Rand1', 'Rand2']
local = [9.0, 15.0, 17.0, 4.0, 4.0]
exhaust = [6.0, 12.0, 11.0, 0.0, 0.0]
ours = [7.0, 12.0, 11.0, 0.0, 0.0]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 0.28, local, 0.2, label='Local')
rects2 = ax.bar(x, exhaust, 0.2, label='Exhaust')
rects3 = ax.bar(x + 0.28, ours, 0.2, label='Greedy')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Cost')
ax.set_title('Costs by Program and Search Strategy')
ax.set_xticks(x)
print(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.savefig(r'C:\Users\maxwell\Desktop\output_img')
plt.show()