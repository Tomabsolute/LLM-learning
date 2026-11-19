losses = [0.8, 0.7, 0.75, 0.72, 0.71, 0.69, 0.70]

def early_stopping(losses, patience=3, tol=0.001):
    best_loss = losses[0]
    count = 0
    for i in range(1,len(losses)):
        current_loss = losses[i]
        if best_loss-current_loss > tol:
            best_loss = current_loss
            count = 0
        else:
            count+=1
            if count == patience:
                return i+1
    return None
        

print(early_stopping(losses))