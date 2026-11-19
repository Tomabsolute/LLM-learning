import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polynomial_regression import polynomial_cv,polynomial_fit,polynomial_predict
from step_functions import step_cv,step_fit,step_predict
from regression_splines import spline_cv,cubic_spline_fit,cubic_spline_predict
from regression_nature_splines import natural_spline_cv,natural_spline_fit,natural_spline_predict,natural_spline_features
from smoothing_splines import smoothing_spline_fit
from local_regression import local_regression_cv,local_linear_predict
from generalized_additive_models import gam_cv,gam_fit,gam_predict

rng = np.random.default_rng(42)

n = 240
x1 = rng.uniform(-3, 3, n)
x2 = rng.uniform(-3, 3, n)
noise = rng.normal(0, 0.4, n)

y_uni = 2 + 2*np.sin(x1) + noise
y_gam = 2 + 2*np.sin(x1) + 0.5*x2**2 + noise

X_gam = np.column_stack([x1, x2])

idx = rng.permutation(n)
train_idx = idx[:192]
test_idx = idx[192:]

X1_train = x1[train_idx]
X1_test = x1[test_idx]
Y1_train = y_uni[train_idx]
Y1_test = y_uni[test_idx]

X2_train = X_gam[train_idx]
X2_test = X_gam[test_idx]
Y2_train = y_gam[train_idx]
Y2_test = y_gam[test_idx]

# 单变量
Models = ["Polynomial Regression","Step Functions","Cubic Regression Splines","Natural Cubic Splines","Penalized Natural Splines","Local Linear Regression"]
Config = []
CV_MSE = []
TEST_MSE = []
MAE = []
R2 = []
# Polynomial Regression
degrees = list(range(1,11))
df,best_degree = polynomial_cv(X1_train,Y1_train,degrees)
cv_mse = df["cv_mse"].min()
beta = polynomial_fit(X1_train,Y1_train,best_degree)
y_pred = polynomial_predict(X1_test,beta)
test_mse = np.mean((Y1_test-y_pred)**2)
mae = np.mean(np.abs(Y1_test-y_pred))
rss = np.sum((Y1_test-y_pred)**2)
tss = np.sum((Y1_test-np.mean(Y1_test))**2)
r2 = 1-rss/tss
Config.append(best_degree)
TEST_MSE.append(test_mse)
MAE.append(mae)
R2.append(r2)
CV_MSE.append(cv_mse)

# Step Functions
K_values = list(range(2,13))
df,best_K = step_cv(X1_train,Y1_train,K_values)
cv_mse = df["cv_mse"].min()
CV_MSE.append(cv_mse)
beta = step_fit(X1_train,Y1_train,best_K)
y_pred = step_predict(X1_test,beta,best_K)
test_mse = np.mean((Y1_test-y_pred)**2)
mae = np.mean(np.abs(Y1_test-y_pred))
rss = np.sum((Y1_test-y_pred)**2)
tss = np.sum((Y1_test-np.mean(Y1_test))**2)
r2 = 1-rss/tss
Config.append(best_K)
TEST_MSE.append(test_mse)
MAE.append(mae)
R2.append(r2)

# Cubic Regression Splines
K_values = np.arange(1,7)
best_K,df = spline_cv(X1_train,Y1_train,K_values)
cv_mse = df["cv_mse"].min()
CV_MSE.append(cv_mse)
knots = np.linspace(-3,3,best_K+2)[1:-1]
beta = cubic_spline_fit(X1_train,Y1_train,knots)
y_pred = cubic_spline_predict(X1_test,beta,knots)
test_mse = np.mean((Y1_test-y_pred)**2)
mae = np.mean(np.abs(Y1_test-y_pred))
rss = np.sum((Y1_test-y_pred)**2)
tss = np.sum((Y1_test-np.mean(Y1_test))**2)
r2 = 1-rss/tss
Config.append(best_K)
TEST_MSE.append(test_mse)
MAE.append(mae)
R2.append(r2)

# Natural Cubic Splines
K_values = list(range(1,7))
best_K,df = natural_spline_cv(X1_train,Y1_train,K_values)
cv_mse = df["cv_mse"].min()
CV_MSE.append(cv_mse)
knots = np.linspace(-3,3,best_K+2)[1:-1]
beta = natural_spline_fit(X1_train,Y1_train,knots)
y_pred = natural_spline_predict(X1_test,beta,knots)
test_mse = np.mean((Y1_test-y_pred)**2)
mae = np.mean(np.abs(Y1_test-y_pred))
rss = np.sum((Y1_test-y_pred)**2)
tss = np.sum((Y1_test-np.mean(Y1_test))**2)
r2 = 1-rss/tss
Config.append(best_K)
TEST_MSE.append(test_mse)
MAE.append(mae)
R2.append(r2)
natural_df = best_K + 2


# Penalized Natural Splines
lambdas = [0,0.01,0.1,1,10,100]
knots = np.linspace(-3, 3, 14)[1:-1]
MSE = []
X1 = np.array_split(X1_train,5)
Y1 = np.array_split(Y1_train,5)
for lam in lambdas:
    MSE_1 = []
    for i in range(5):
        X_train = np.concatenate(X1[:i]+X1[i+1:])
        X_val = X1[i]
        Y_train = np.concatenate(Y1[:i]+Y1[i+1:])
        Y_val = Y1[i]
        beta,Omega,S = smoothing_spline_fit(X_train,Y_train,knots,lam)
        y_pred = natural_spline_features(X_val,knots)@beta
        mse = np.mean((Y_val-y_pred)**2)
        MSE_1.append(mse)
    MSE.append(np.mean(MSE_1))
idx = np.argmin(MSE)
cv_mse = MSE[idx]
CV_MSE.append(cv_mse)
best_lam = lambdas[idx]
beta,Omega,S = smoothing_spline_fit(X1_train,Y1_train,knots,best_lam)
y_pred = natural_spline_features(X1_test,knots)@beta
test_mse = np.mean((Y1_test-y_pred)**2)
mae = np.mean(np.abs(Y1_test-y_pred))
rss = np.sum((Y1_test-y_pred)**2)
tss = np.sum((Y1_test-np.mean(Y1_test))**2)
r2 = 1-rss/tss
Config.append(best_lam)
TEST_MSE.append(test_mse)
MAE.append(mae)
R2.append(r2)
penalized_df = np.trace(S)

# Local Linear Regression
fracs = [0.1,0.2,0.3,0.4,0.6,0.8]
best_frac,df = local_regression_cv(X1_train,Y1_train,fracs)
cv_mse = df["cv_mse"].min()
CV_MSE.append(cv_mse)
y_pred = local_linear_predict(X1_train,Y1_train,X1_test,best_frac)
test_mse = np.mean((Y1_test-y_pred)**2)
mae = np.mean(np.abs(Y1_test-y_pred))
rss = np.sum((Y1_test-y_pred)**2)
tss = np.sum((Y1_test-np.mean(Y1_test))**2)
r2 = 1-rss/tss
Config.append(best_frac)
TEST_MSE.append(test_mse)
MAE.append(mae)
R2.append(r2)

df = pd.DataFrame(dict(Model=Models,Config=Config,CV_MSE=CV_MSE,Test_MSE = TEST_MSE,MAE=MAE,R2=R2)).sort_values("CV_MSE")
top3 = df.head(3)
print(df)

# GAM
K_values = np.arange(1,7)
best_K,df = gam_cv(X2_train, Y2_train, K_values)
knots = np.linspace(-3,3,best_K+2)[1:-1]
beta = gam_fit(X2_train,Y2_train,knots,knots)
y_pred = gam_predict(X2_test,beta,knots,knots)
test_mse = np.mean((y_pred - Y2_test)**2)
mae = np.mean(np.abs(y_pred-Y2_test))
rss = np.sum((y_pred - Y2_test)**2)
tss = np.sum((Y2_test - np.mean(Y2_test))**2)
r2 = 1-rss/tss
print("GAM's Test MSE:",test_mse)
print("GAM's MAE:",mae)
print("GAM's R2:",r2)

print("natural_df:",natural_df)
print("penalized_df:",penalized_df)

x_grid = np.linspace(-3, 3, 300)
beta_1 = beta[1:best_K+2]
beta_2 = beta[best_K+2:]
X_1 = natural_spline_features(x_grid,knots)[:,1:]

y_pred1 = X_1@beta_1
y_pred2 = X_1@beta_2

plt.figure()
plt.plot(x_grid,y_pred1,label = "f_1(x)")
plt.plot(x_grid,y_pred2,label = "f_2(x)")
plt.legend()
plt.show()

x_grid = np.linspace(-3, 3, 300)

plt.figure()
plt.scatter(
    X1_train, Y1_train,
    s=8, alpha=0.4, label="Training data"
)

for row in top3.itertuples(index=False):
    name = row.Model
    config = row.Config

    if name == "Polynomial Regression":
        beta = polynomial_fit(
            X1_train, Y1_train, int(config)
        )
        y_curve = polynomial_predict(x_grid, beta)

    elif name == "Natural Cubic Splines":
        knots = np.linspace(
            -3, 3, int(config)+2
        )[1:-1]
        beta = natural_spline_fit(
            X1_train, Y1_train, knots
        )
        y_curve = natural_spline_predict(
            x_grid, beta, knots
        )

    elif name == "Cubic Regression Splines":
        knots = np.linspace(
            -3, 3, int(config)+2
        )[1:-1]
        beta = cubic_spline_fit(
            X1_train, Y1_train, knots
        )
        y_curve = cubic_spline_predict(
            x_grid, beta, knots
        )

    elif name == "Penalized Natural Splines":
        knots = np.linspace(-3, 3, 14)[1:-1]
        beta,Omega,S = smoothing_spline_fit(X1_train,Y1_train,knots,float(config))
        y_curve = natural_spline_features(x_grid,knots)@beta

    else:
        raise ValueError(f"未处理的模型: {name}")

    plt.plot(x_grid, y_curve, label=name)

plt.legend()
plt.show()

lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]
knots = np.linspace(-3, 3, 14)[1:-1]
X = natural_spline_features(x1,knots)
RSS = []
DF = []
plt.figure()
for lam in lambdas:
    beta,Omega,S = smoothing_spline_fit(X1_train,Y1_train,knots,lam)
    df_lam = np.trace(S)
    DF.append(df_lam)
plt.plot(lambdas, DF, marker="o")
plt.xscale("symlog", linthresh=0.01)
plt.xlabel("lambda")
plt.ylabel("Effective df")
plt.title("Model complexity")
plt.show()




