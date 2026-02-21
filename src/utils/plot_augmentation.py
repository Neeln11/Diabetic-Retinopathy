
import matplotlib.pyplot as plt
import numpy as np

# Original Counts (approx from 16.6k dataset)
# 0: 8200
# 1: 1022
# 2: 5632
# 3: 513
# 4: 1272
original = [8200, 1022, 5632, 513, 1272]

# Balanced counts (Target)
balanced = [8200, 8200, 8200, 8200, 8200]

labels = ['0 (No DR)', '1 (Mild)', '2 (Mod)', '3 (Sev)', '4 (Prolif)']
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, original, width, label='Original Distribution', color='#e74c3c')
rects2 = ax.bar(x + width/2, balanced, width, label='Effective Training (Augmented)', color='#2ecc71')

ax.set_ylabel('Number of Samples per Epoch')
ax.set_title('Impact of Class-Balanced Augmentation Strategy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('docs/Data_Augmentation_Distribution.png')
print("Chart generated: docs/Data_Augmentation_Distribution.png")
