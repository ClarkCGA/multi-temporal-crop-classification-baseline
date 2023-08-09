import torch
from custom_loss_functions import *


def validate_one_epoch(valData, model, criterion, device, val_loss=[]):
    """
        Evaluate the model on separate Landsat scenes.
        Params:
            valData (DataLoader object) -- Batches of image chips from PyTorch custom dataset(AquacultureData)
            model -- Choice of segmentation Model.
            criterion -- Chosen function to calculate loss over validation samples.
            device (str): Either 'cuda' or 'cpu'.
            val_loss (empty list): To record average loss for each epoch
    """

    model.eval()

    # mini batch iteration
    eval_epoch_loss = 0
    num_val_batches = len(valData)

    with torch.no_grad():
        
        for img_chips, labels in valData:
            
            img = img_chips.to(device)
            label = labels.to(device)

            pred = model(img)

            loss = eval(criterion)(pred, label)
            eval_epoch_loss += loss.item()

    print('validation loss: {:.4f}'.format(eval_epoch_loss / num_val_batches))

    if val_loss is not None:
        val_loss.append(float(eval_epoch_loss / num_val_batches))