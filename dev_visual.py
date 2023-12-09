import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 주어진 confusion matrix
confusion_matrix_data = np.array([[360, 230],
                                  [16, 139]])

# confusion matrix 시각화
plt.figure(figsize=(6, 4))
sns.set(font_scale=1.2)  # 폰트 크기 설정
sns.heatmap(confusion_matrix_data, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Positive", "Negative"], yticklabels=["Positive", "Negative"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
