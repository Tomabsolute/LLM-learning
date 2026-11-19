import numpy as np

y_true = np.array([
    1, 0, 1, 1, 0,
    0, 1, 0, 1, 0
])

y_prob = np.array([
    0.95, 0.80, 0.70, 0.60, 0.55,
    0.45, 0.40, 0.30, 0.20, 0.10
])


def classification_metrics(y_true, y_prob, threshold=0.5):
    if (len(y_true) != len(y_prob)) or (not np.all((0<=y_prob) & (y_prob<=1))) or (not 0<=threshold<=1):
        raise ValueError
    y_hat = np.where(y_prob>=threshold,1,0)
    tp = np.sum((y_true==1) & (y_hat == 1))
    fp = np.sum((y_true==0) &( y_hat == 1))
    tn = np.sum((y_true==0) & (y_hat == 0))
    fn = np.sum((y_true==1) & (y_hat == 0))
    accuracy = (tp+tn)/(tp+fp+tn+fn) if tp+fp+tn+fn != 0 else 0
    precision = tp/(tp+fp)if tp+fp != 0 else 0
    recall = tp/(tp+fn)if tp+fn != 0 else 0
    specificity = tn/(tn+fp)if tn+fp != 0 else 0
    f1 = 2*precision*recall/(precision+recall)if precision+recall != 0 else 0
    return dict(tp=tp,fp=fp,tn=tn,fn=fn,accuracy=accuracy,precision=precision,recall=recall,specificity=specificity,f1=f1)

result = classification_metrics(
    y_true, y_prob, threshold=0.5
)

print(result)

for threshold in [0.3, 0.5, 0.7, 0.9]:
    result = classification_metrics(
        y_true, y_prob, threshold
    )
    print(threshold,result["recall"],result["precision"])