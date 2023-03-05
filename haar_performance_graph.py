import matplotlib.pyplot as plt
import numpy as np

# Define the data
labels = ["Default", "Custom 1", "Custom 2", "Custom 3"]
true_positives = [5131, 4237, 5641, 5171]
false_positives = [77, 0, 188, 2]
true_negatives = [2657, 2734, 2546, 2732]
false_negatives = [1059, 1953, 549, 1019]
precision = [0.99, 1.0, 0.97, 1.0]
recall = [0.83, 0.68, 0.91, 0.84]
f1_score = [0.98, 0.81, 0.94, 0.91]

# Set up the plot
x = np.arange(len(labels))
width = 0.2
fig, ax = plt.subplots()
rects1 = ax.bar(x - 3 * width / 2, true_positives, width, label="True Positives")
rects2 = ax.bar(x - width / 2, false_positives, width, label="False Positives")
rects3 = ax.bar(x + width / 2, true_negatives, width, label="True Negatives")
rects4 = ax.bar(x + 3 * width / 2, false_negatives, width, label="False Negatives")

# Add labels, title, and legend
ax.set_xlabel("Algorithm")
ax.set_ylabel("Count")
ax.set_title("Performance Metrics by Algorithm")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add the text labels for precision, recall, and f1 score
for i, rect in enumerate(rects1):
    ax.text(
        rect.get_x() + rect.get_width() / 2.0,
        1.02 * rect.get_height(),
        f"Precision: {precision[i]:.2f}\nRecall: {recall[i]:.2f}\nF1 Score: {f1_score[i]:.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

# Show the plot
plt.tight_layout()
plt.show()
