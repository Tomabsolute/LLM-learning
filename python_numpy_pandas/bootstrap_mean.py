import numpy as np

data = np.array([
    12.5, 13.2, 11.8, 14.1, 12.9,
    13.7, 11.5, 14.3, 12.1, 13.0
])

def bootstrap_mean(data, B=1000, seed=42):
    if data.ndim != 1 or data.size == 0 or (not B >= 2):
        raise ValueError
    rng = np.random.default_rng(seed)
    original_mean = np.mean(data)
    samples = rng.choice(data,size = (B,len(data)),replace = True)
    mean = np.mean(samples,axis = 1)
    bootstrap_mean = np.mean(mean)
    bootstrap_se = np.std(mean,ddof = 1)
    lower, upper = np.percentile(mean, [2.5, 97.5])
    ci = [lower,upper]
    return dict(original_mean = original_mean,bootstrap_mean = bootstrap_mean,bootstrap_se = bootstrap_se,ci = ci,bootstrap_samples = mean)

result = bootstrap_mean(data)

print("Original mean:", result["original_mean"])
print("Bootstrap mean:", result["bootstrap_mean"])
print("Bootstrap SE:", result["bootstrap_se"])
print("95% CI:", result["ci"])