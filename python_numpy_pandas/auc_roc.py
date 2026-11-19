import numpy as np
import matplotlib.pyplot as plt

y_true = np.array([
    1, 0, 1, 1, 0,
    0, 1, 0, 1, 0
])

y_prob = np.array([
    0.95, 0.80, 0.70, 0.60, 0.55,
    0.45, 0.40, 0.30, 0.20, 0.10
])

def roc_curve_manual(y_true, y_prob):
    p = y_prob
    y_true_1 = y_true

    thresholds = np.r_[np.inf, np.unique(y_prob)[::-1]]
    FPR = []
    TPR = []
    for i in thresholds:
        tp = np.sum((y_true_1 == 1)&(p >= i))
        fp = np.sum((y_true_1 == 0)&(p >= i))
        tn = np.sum((y_true_1 == 0)&(p < i))
        fn = np.sum((y_true_1 == 1)&(p < i))
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        FPR.append(fpr)
        TPR.append(tpr)
    return np.array(FPR),np.array(TPR),thresholds


def auc_manual(fpr, tpr):
    result = 0
    for i in range(len(fpr)-1):
        result+=(tpr[i]+tpr[i+1])/2*(fpr[i+1]-fpr[i])
    return result
    
fpr, tpr, thresholds = roc_curve_manual(y_true, y_prob)
auc = auc_manual(fpr, tpr)

print("FPR:", fpr)
print("TPR:", tpr)
print("Thresholds:", thresholds)
print("AUC:", auc)

plt.figure()
plt.plot(np.linspace(0,1,100),np.linspace(0,1,100),color = "yellow",ls = "--")
plt.plot(fpr,tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

y_true2 = np.array([1, 0, 1, 0])
y_prob2 = np.array([0.8, 0.8, 0.3, 0.2])

fpr2, tpr2, thresholds2 = roc_curve_manual(
    y_true2, y_prob2
)

print(fpr2)
print(tpr2)
print(auc_manual(fpr2, tpr2))