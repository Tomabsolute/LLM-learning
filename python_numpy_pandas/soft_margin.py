import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

rng = np.random.default_rng(42)

n = 150

X = rng.normal(size=(n, 2))

score = (
    X[:, 0]
    + X[:, 1]
    + rng.normal(0, 1.2, n)
)

y = np.where(score > 0, 1, -1)



def soft_margin_fit(X,y,C=1.0):
    if X.ndim != 2:
        raise ValueError
    if X.shape[1] != 2:
        raise ValueError
    if y.shape != (X.shape[0],):
        raise ValueError
    if not np.array_equal(np.unique(y),[-1,1]):
        raise ValueError

    def objective(params):
        beta = params[1:X.shape[1]+1]
        xi = params[X.shape[1]+1:]
        return np.dot(beta,beta)/2+C*np.sum(xi)

    def objective_gradient(params):
        return np.concatenate(([0],params[1:X.shape[1]+1],[C]*len(X)))

    def constraint_1(params):
        return y*(params[0]+X@params[1:X.shape[1]+1])-1+params[X.shape[1]+1:]

    def constraint_1_gradient(params):
        return np.column_stack([y,y[:,None]*X,np.identity(len(X))])

    def constraint_2(params):
        return params[X.shape[1]+1:]

    def constraint_2_gradient(params):
        return np.column_stack([np.zeros((len(X),X.shape[1]+1)),np.identity(len(X))])

    result = minimize(
        fun = objective,
        x0 = np.zeros(len(X)+X.shape[1]+1),
        jac = objective_gradient,
        method = "SLSQP",
        constraints = [
            {
                "type":"ineq",
                "fun":constraint_1,
                "jac":constraint_1_gradient
            },
            {
                "type":"ineq",
                "fun":constraint_2,
                "jac":constraint_2_gradient                
            }
        ],
        options = {
            "maxiter":1000,
            "ftol":1e-6
        }
    )

    if not result.success:
        raise RuntimeError(f"优化失败{C}")

    beta_0 = result.x[0]
    beta = result.x[1:X.shape[1]+1]
    xi = result.x[X.shape[1]+1:]
    support_vectors = X[y*(beta_0+X@beta)-1 <= 1e-5]
    margin = 1/np.linalg.norm(beta)
    return beta_0,beta,xi,support_vectors,margin

def soft_margin_predict(X_new,model):
    beta_0,beta,xi,support_vectors,margin = model
    y_pred = np.where(beta_0+X_new@beta>=0,1,-1)
    return y_pred

if __name__ == "__main__":
    C_values = [0.01, 0.1, 1, 10]
    Models = []
    Accuracy = []
    num_support_vectors = []
    Margin = []
    for C in C_values:
        model = soft_margin_fit(X,y,C)
        beta_0,beta,xi,support_vectors,margin = model
        y_pred = soft_margin_predict(X,model)
        accuracy = np.sum(y_pred == y)/len(y)
        Models.append(model)
        Accuracy.append(accuracy)
        num_support_vectors.append(len(support_vectors))
        Margin.append(margin)
    df = pd.DataFrame({
                        "C":C_values,
                       "Accuracy":Accuracy,
                       "num_support_vectors":num_support_vectors,
                       "Margin":Margin
                       })
    print(df)
    

    plt.figure(figsize=(7, 6))
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class +1")
    plt.scatter(X[y == -1, 0], X[y == -1, 1], label="Class -1")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Linearly Non-separable Data")
    plt.legend()
    plt.show()

    iterator = iter(Models)
    iterator_C = iter(C_values)
    fig,ax = plt.subplots(2,2)
    padding = 0.5
    x_limit = (np.min(X[:,0])-padding,np.max(X[:,0]+padding))
    y_limit = (np.min(X[:,1])-padding,np.max(X[:,1]+padding))
    xx,yy = np.meshgrid(
        np.linspace(*x_limit,400),
        np.linspace(*y_limit,400)
    )
    for i in range(2):
        for j in range(2):
            beta_0,beta,xi,support_vectors,margin = next(iterator)
            scores = beta_0+xx*beta[0]+yy*beta[1]
            ax[i][j].contour(xx,yy,scores,
                           levels = [-1,0,1],
                           colors = "black",
                           linestyles = ["--","-","--"]
                           )
            ax[i][j].plot([],[],ls = "--",label = "Margins: f(x) = +/-1")
            ax[i][j].plot([],[],ls = "-",label = "Hyperplane: f(x) = 0")
            ax[i][j].set(xlim = x_limit,ylim = y_limit,xlabel = "x1",ylabel = "x2",title = f"{next(iterator_C)}")
            ax[i][j].legend()
            ax[i][j].grid(alpha = 0.2)
    fig.tight_layout()
    plt.show()