from torch.autograd import Variable
from custom_loss_functions import *


def train_one_epoch(trainData, model, criterion, optimizer, scheduler, lr_policy, device, train_loss=[]):
    r"""
    Train the model.
    Arguments:
            trainData (DataLoader object) -- Batches of image chips from PyTorch custom dataset(AquacultureData)
            model (initialized model) -- Choice of segmentation Model to train.
            criterion -- Chosen function to calculate loss over training samples.
            optimizer -- Chosen function for optimization.
            scheduler -- Update policy for learning rate decay.
            lr_policy (str) -- Learning rate decade policy.
            device --(str) Either 'cuda' or 'cpu'.
            train_loss -- (empty list) To record average training loss for each epoch.
            
    """
    model.train()

    epoch_loss = 0
    num_train_batches = len(trainData)

    for img_chips, labels in trainData:

        img = img_chips.to(device)
        label = labels.to(device)

        model_out = model(img)

        loss = eval(criterion)(model_out, label)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_policy == "CyclicLR":
            scheduler.step()

    print('train loss:{}'.format(epoch_loss / num_train_batches))

    if lr_policy == "CyclicLR":
        print("LR: {}".format(scheduler.get_last_lr()))

    if train_loss is not None:
        train_loss.append(float(epoch_loss / num_train_batches))