import matplotlib.pyplot as plt
import torch

# make a plot of AUS with errors as a function of dataset size
AUSs = [90.0, 91.5, 92.27, 94.03, 92.89]
AUSs_std = [4.07, 4.92, 2.94, 2.34, 2.82]
sizes = [2000, 4000, 6000, 8000, 10000]

plt.plot(sizes, AUSs, '-o')
plt.fill_between(sizes, [x - y for x, y in zip(AUSs, AUSs_std)], [x + y for x, y in zip(AUSs, AUSs_std)], alpha=0.2)
#y range axis range
plt.ylim(50, 100)
#set visible x ticks
plt.xticks(sizes)
plt.xlabel('Dataset size')
plt.ylabel('AUS')
plt.savefig('AUSs_vs_size.png')

# #plot histogram of the classified classes
# import pandas as pd
# outputs = torch.randint(1000)
# #read from csv
# o = torch.cat(outputs).to(opt.device)
# o = torch.argmax(o, dim = 1)
# idx = torch.zeros(10000)
# for j in range(10):   
#     idx = torch.logical_or(idx, o==j*10)
# AUSs = []
# def AUS(a_t, a_f):  
#     a_or = 0.7756
#     aus=(1-(a_or-a_t))/(1+abs(a_f))
#     return aus

# for i in range(10):
#     csv = pd.read_csv(f"/home/lsabetta/Documents/trick_distill/src/out/CR/cifar100/dfs/Mahalanobis_seed_42_class_{i*10}.csv")
#     forget = csv["forget_test_accuracy"][0]
#     retain = csv["retain_test_accuracy"][0]
#     AUSs.append(AUS(retain,forget))

# counts, _ , _= plt.hist(o[idx], bins = 10)
# plt.plot(counts, AUSs, ls = "", marker = ".")
# plt.xlabel("N forget samples")
# plt.ylabel("AUS")