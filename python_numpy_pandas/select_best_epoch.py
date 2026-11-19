train_losses = [0.80, 0.60, 0.40, 0.30, 0.20, 0.10, 0.05]
val_losses = [0.85, 0.65, 0.50, 0.45, 0.48, 0.55, 0.65]

def select_best_epoch(train_losses, val_losses):
    if (len(train_losses) != len(val_losses)) or (not train_losses):
        raise ValueError
    min_loss = val_losses[0]
    min_arg = 0 #epoch = min_arg+1
    for i in range(1,len(val_losses)):
        current_loss = val_losses[i]
        if current_loss < min_loss:
            min_loss = current_loss
            min_arg = i
    return (min_arg+1,min_loss,train_losses[min_arg])
        

print(select_best_epoch(train_losses, val_losses))