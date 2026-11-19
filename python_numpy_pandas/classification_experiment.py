import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

X0 = rng.normal(
    loc=[0, 0],
    scale=[1, 1],
    size=(100, 2)
)

X1 = rng.normal(
    loc=[1.5, 1.5],
    scale=[1, 1],
    size=(100, 2)
)

X = np.vstack([X0, X1])
y = np.array([0]*100 + [1]*100)

def assessment(y,y_hat,y_prob):
    tp = np.sum((y==1)&(y_hat==1))
    fp = np.sum((y==0)&(y_hat==1))
    tn = np.sum((y==0)&(y_hat==0))
    fn = np.sum((y==1)&(y_hat==0))
    accuracy = (tp+tn)/(tp+tn+fp+fn) if tp+tn+fp+fn != 0 else 0
    precision = tp/(tp+fp) if tp+fp != 0 else 0
    recall = tp/(tp+fn) if tp+fn != 0 else 0
    F1_score = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0

    index = np.argsort(y_prob)
    y_prob_1 = y_prob[index]
    y_1 = y[index]
    thresholds = np.r_[np.inf,np.unique(y_prob_1)[::-1]]
    FPR = []
    TPR = []
    for i in thresholds:
        y_hat_1 = np.where(y_prob_1>=i,1,0)
        tp = np.sum((y_1==1)&(y_hat_1==1))
        fp = np.sum((y_1==0)&(y_hat_1==1))
        tn = np.sum((y_1==0)&(y_hat_1==0))
        fn = np.sum((y_1==1)&(y_hat_1==0))
        fpr = fp/(fp+tn)
        tpr = tp/(tp+fn)
        FPR.append(fpr)
        TPR.append(tpr)
    auc = 0
    for i in range(len(FPR)-1):
        auc+= (TPR[i]+TPR[i+1])*(FPR[i+1]-FPR[i])/2
    return dict(accuracy = accuracy,precision = precision,recall = recall,F1_score = F1_score,auc = auc,FPR = FPR,TPR = TPR)

def sigmoid(z):
    z_1 = np.exp(-np.abs(z))
    p = np.where(z>=0,1/(1+z_1),z_1/(1+z_1))
    return p

def logistic_regression(X,y,epochs = 1000,learning_rate = 0.2):
    beta = np.zeros(X.shape[1])
    b = 0
    for i in range(epochs):
        z = X@beta+b
        p = sigmoid(z)
        L_beta = -np.mean((y-p)*X.T,axis=1)
        L_b = -np.mean(y-p)
        beta = beta-learning_rate*L_beta
        b = b-learning_rate*L_b
    return beta,b

def predict_logistic(X,beta,b,threshold = 0.5):
    y_prob = sigmoid(X@beta+b)
    y_hat = np.where(y_prob>=threshold,1,0)
    return y_prob,y_hat

def predict_gnb(X,p,classes,mean,var):
    result = []
    for i,c in enumerate(classes):
        p_k = p[i]
        mean_k = mean[i]
        var_k = var[i]
        delta = np.log(p_k)+np.sum(-np.log(var_k)/2)+np.sum(-(X-mean_k)**2/(2*var_k),axis = 1)
        result.append(delta)
    result_1 = np.column_stack(result)
    index = np.argmax(result_1,axis = 1)
    return classes[index],sigmoid(-result[0]+result[1])
    
def predict_knn(X_train,y_train,X_val,k=5):
    d = np.sqrt(np.sum((X_val[:,None,:]-X_train[None,:,:])**2,axis = 2))
    index = np.argsort(d,axis = 1)[:,:k]
    class_1 = y_train[index]
    y_prob = np.mean(class_1 == 1,axis = 1)
    y_hat = []
    for i in class_1:
        classes,counts = np.unique(i,return_counts = True)
        idx = np.argsort(counts)[-1]
        cla = classes[idx]
        y_hat.append(cla)
    return np.array(y_hat),y_prob

def classification_experiment(X, y, seed=42):
    rng = np.random.default_rng(seed)
    index = rng.permutation(len(X))
    X_1 = X[index]
    y_1 = y[index]
    X_train = X_1[:int(0.7*len(X_1))]
    X_val = X_1[int(0.7*len(X_1)):]
    y_train = y_1[:int(0.7*len(y_1))]
    y_val = y_1[int(0.7*len(y_1)):]

    # logistic regression
    beta,b = logistic_regression(X_train,y_train)
    y_prob,y_hat = predict_logistic(X_val,beta,b)
    dic_1 = assessment(y_val,y_hat,y_prob)

    # Gaussian Naive Bayes
    classes,counts = np.unique(y_train,return_counts = 1)
    p = counts/np.sum(counts)
    X_mean = np.array([np.mean(X_train[y_train==c],axis = 0) for c in classes])
    X_var = np.array([np.var(X_train[y_train==c],axis = 0) for c in classes])
    y_hat,y_score = predict_gnb(X_val,p,classes,X_mean,X_var)
    dic_2 = assessment(y_val,y_hat,y_score)

    # KNN
    y_hat,y_prob = predict_knn(X_train,y_train,X_val,k=5)
    dic_3 = assessment(y_val,y_hat,y_prob)

    df = pd.DataFrame([dic_1,dic_2,dic_3],index = ["Logistic Regression","Gaussion Naive Bayes","KNN"],columns=["accuracy","precision","recall","F1_score","auc"])
    df = df.sort_values(by = "auc",ascending = False)
    return df,[dic_1["FPR"], dic_2["FPR"], dic_3["FPR"]],[dic_1["TPR"], dic_2["TPR"], dic_3["TPR"]]

result,FPR,TPR = classification_experiment(X, y)
print(result)

plt.figure()
plt.plot(FPR[0],TPR[0],label = "Logistic Regression")
plt.plot(FPR[1],TPR[1],label = "Gaussion Naive Bayes")
plt.plot(FPR[2],TPR[2],label = "KNN")
plt.plot([0,1],[0,1],ls ="--",label = "random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()