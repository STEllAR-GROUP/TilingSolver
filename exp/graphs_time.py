import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['LR', 'PCA', 'Power Set', 'Rand1', 'Rand2']
local = [0.0013, 0.0019, 0.0067, 0.0036, 0.0013]
exhaust = [0.6818, 1158.9374, 2367.5178, 1.9240, 0.0625]
ours = [0.0096, 0.0407, 0.0292, 0.0128, 0.0114]

x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 0.1, local, width, label='Local')
rects2 = ax.bar(x, exhaust, width, label='Exhaust')
rects3 = ax.bar(x + 0.1, exhaust, width, label='Ours')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time')
ax.set_title('Time by Program and Search Strategy')
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