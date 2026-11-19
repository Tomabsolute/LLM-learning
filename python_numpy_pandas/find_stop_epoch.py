losses = [1.0, 0.8, 0.7, 0.6995, 0.6991, 0.6990, 0.6980]

tol = 0.001
patience = 2

def find_stop_epoch(losses, tol, patience):
    count = 0
    for i in range(1,len(losses)):
        if losses[i-1]-losses[i] < tol:
            count+=1
            if count == patience:
                return i+1
        else:
            count = 0
    return None

print(find_stop_epoch(losses, tol, patience))