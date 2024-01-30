import torch
import torchvision
import torch.nn as nn
import torch.optim as optim


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import numpy as np
import pickle as pk

def compute_accuracy(net, loader, opt):
    correct = 0
    total = 0
    net.eval()  # Set the model to evaluation mode
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(opt.device), labels.to(opt.device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store all labels and predictions for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    return 100 * correct / total,cm

def obtain_MIA_data(model,loader,opt,train=True):
    model.eval()  # Set the model to evaluation mode
    
    classes = []
    predictions = []

    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(opt.device), labels.to(opt.device)
            outputs = model(inputs)
            
            # Store all labels and predictions for confusion matrix
            classes.extend(labels.detach().cpu().numpy())
            predictions.extend(outputs.detach().cpu().numpy())

    classes = np.array(classes)
    predictions = np.array(predictions)
    if train:
        case = np.ones_like(classes)
    else:
        case = np.zeros_like(classes)
    return classes,predictions,case


def train_model(model,optimizer,criterion,scheduler,trainloader,testloader,opt,args,plot_name=None):
    if args.test:
        print('In test: ')
        model.load_state_dict(torch.load(opt.model_weights_name+'.pth'))
        acc_test,cm = compute_accuracy(model, testloader,opt)
        print(f'test_acc: {acc_test}')
        if not(plot_name is None):
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d')
            plt.xlabel('Predicted')
            plt.ylabel('Truth')
            plt.title(f'test_acc: {acc_test}')
            plt.savefig(f'{plot_name}_best_res.png')  # Save the plot
            plt.close()

    else:
        acc_best = 0
        for epoch in range(opt.epochs):  # loop over the dataset multiple times
            
            model.train()
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(opt.device), labels.to(opt.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            scheduler.step()
            acc_train,_ = compute_accuracy(model, trainloader,opt)
            acc_test,cm = compute_accuracy(model,testloader ,opt)
            print(f'epoch_{epoch}, train_acc: {round(acc_train,3)}, test_acc: {round(acc_test,3)}, loss: {round(running_loss/i,3)}')

            if acc_test > acc_best:
                acc_best = acc_test
                torch.save(model.state_dict(), opt.model_weights_name+'.pth')
                    # Plot confusion matrix
                if not(plot_name is None):
                    plt.figure(figsize=(10, 7))
                    sns.heatmap(cm, annot=True, fmt='d')
                    plt.xlabel('Predicted')
                    plt.ylabel('Truth')
                    plt.savefig(f'{plot_name}_{epoch}_.png')  # Save the plot
                    plt.close()

        print('Finished Training')
        ##### load best model  ###
        model.load_state_dict(torch.load(opt.model_weights_name+'.pth'))
        return model