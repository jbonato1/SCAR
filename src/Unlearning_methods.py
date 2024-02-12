import torch
import torchvision
from torch import nn 
from torch import optim
from torch.nn import functional as F
from opts import OPT as opt
import pickle
from tqdm import tqdm
from torchattacks import PGD
from utils import accuracy
import time
from copy import deepcopy


def choose_method(name):
    if name=='FineTuning':
        return FineTuning
    elif name=='NegativeGradient':
        return NegativeGradient
    elif name=='RandomLabels':
        return RandomLabels
    elif name=='DUCK':
        return DUCK
    elif name=='Mahalanobis':
        return Mahalanobis

class BaseMethod:
    def __init__(self, net, retain, forget,test=None):
        self.net = net
        self.retain = retain
        self.forget = forget
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=opt.lr_unlearn, momentum=opt.momentum_unlearn, weight_decay=opt.wd_unlearn)
        self.epochs = opt.epochs_unlearn
        self.target_accuracy = opt.target_accuracy
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.scheduler, gamma=0.5)
        if test is None:
            pass 
        else:
            self.test = test
    def loss_f(self, net, inputs, targets):
        return None

    def run(self):
        self.net.train()
        for _ in tqdm(range(self.epochs)):
            for inputs, targets in self.loader:
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                self.optimizer.zero_grad()
                loss = self.loss_f(inputs, targets)
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                self.net.eval()
                curr_acc = accuracy(self.net, self.forget)
                self.net.train()
                print(f"ACCURACY FORGET SET: {curr_acc:.3f}, target is {self.target_accuracy:.3f}")
                if curr_acc < self.target_accuracy:
                    break

            self.scheduler.step()
            #print('Accuracy: ',self.evalNet())
        self.net.eval()
        return self.net
    
    def evalNet(self):
        #compute model accuracy on self.loader

        self.net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, targets in self.retain:
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            correct2 = 0
            total2 = 0
            for inputs, targets in self.forget:
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total2 += targets.size(0)
                correct2+= (predicted == targets).sum().item()

            if not(self.test is None):
                correct3 = 0
                total3 = 0
                for inputs, targets in self.test:
                    inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                    outputs = self.net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total3 += targets.size(0)
                    correct3+= (predicted == targets).sum().item()
        self.net.train()
        if self.test is None:
            return correct/total,correct2/total2
        else:
            return correct/total,correct2/total2,correct3/total3
    
class FineTuning(BaseMethod):
    def __init__(self, net, retain, forget,test=None,class_to_remove=None):
        super().__init__(net, retain, forget,test=test)
        self.loader = self.retain
        self.target_accuracy=0.0
    
    def loss_f(self, inputs, targets,test=None):
        outputs = self.net(inputs)
        loss = self.criterion(outputs, targets)
        return loss

class RandomLabels(BaseMethod):
    def __init__(self, net, retain, forget,test=None,class_to_remove=None):
        super().__init__(net, retain, forget,test=test)
        self.loader = self.forget
        self.class_to_remove = class_to_remove

        if opt.mode == "CR":
            self.random_possible = torch.tensor([i for i in range(opt.num_classes) if i not in self.class_to_remove]).to(opt.device).to(torch.float32)
        else:
            self.random_possible = torch.tensor([i for i in range(opt.num_classes)]).to(opt.device).to(torch.float32)
    def loss_f(self, inputs, targets):
        outputs = self.net(inputs)
        #create a random label tensor of the same shape as the outputs chosing values from self.possible_labels
        random_labels = self.random_possible[torch.randint(low=0, high=self.random_possible.shape[0], size=targets.shape)].to(torch.int64).to(opt.device)
        loss = self.criterion(outputs, random_labels)
        return loss

class NegativeGradient(BaseMethod):
    def __init__(self, net, retain, forget,test=None):
        super().__init__(net, retain, forget,test=test)
        self.loader = self.forget
    
    def loss_f(self, inputs, targets):
        outputs = self.net(inputs)
        loss = self.criterion(outputs, targets) * (-1)
        return loss

class DUCK(BaseMethod):
    def __init__(self, net, retain, forget,test,class_to_remove=None):
        super().__init__(net, retain, forget, test)
        self.loader = None
        self.class_to_remove = class_to_remove

    def pairwise_cos_dist(self, x, y):
        """Compute pairwise cosine distance between two tensors"""
        x_norm = torch.norm(x, dim=1).unsqueeze(1)
        y_norm = torch.norm(y, dim=1).unsqueeze(1)
        x = x / x_norm
        y = y / y_norm
        return 1 - torch.mm(x, y.transpose(0, 1))


    def run(self):
        """compute embeddings"""
        #lambda1 fgt
        #lambda2 retain


        bbone = torch.nn.Sequential(*(list(self.net.children())[:-1] + [nn.Flatten()]))
        if opt.model == 'AllCNN':
            fc = self.net.classifier
        else:
            fc = self.net.fc
        
        bbone.eval()

 
        # embeddings of retain set
        with torch.no_grad():
            ret_embs=[]
            labs=[]
            cnt=0
            for img_ret, lab_ret in self.retain:
                img_ret, lab_ret = img_ret.to(opt.device), lab_ret.to(opt.device)
                logits_ret = bbone(img_ret)
                ret_embs.append(logits_ret)
                labs.append(lab_ret)
                cnt+=1
                if cnt>=10:
                    break
            ret_embs=torch.cat(ret_embs)
            labs=torch.cat(labs)
        

        # compute distribs from embeddings
        distribs=[]
        for i in range(opt.num_classes):
            if type(self.class_to_remove) is list:
                if i not in self.class_to_remove:
                    distribs.append(ret_embs[labs==i].mean(0))
            else:
                distribs.append(ret_embs[labs==i].mean(0))
        distribs=torch.stack(distribs)
  

        bbone.train(), fc.train()

        optimizer = optim.Adam(self.net.parameters(), lr=opt.lr_unlearn, weight_decay=opt.wd_unlearn)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.scheduler, gamma=0.5)

        init = True
        flag_exit = False
        all_closest_distribs = []
        if opt.dataset == 'tinyImagenet':
            ls = 0.2
        else:
            ls = 0
        criterion = nn.CrossEntropyLoss(label_smoothing=ls)
        

        print('Num batch forget: ',len(self.forget), 'Num batch retain: ',len(self.retain))
        for _ in tqdm(range(opt.epochs_unlearn)):
            for n_batch, (img_fgt, lab_fgt) in enumerate(self.forget):
                for n_batch_ret, (img_ret, lab_ret) in enumerate(self.retain):
                    img_ret, lab_ret,img_fgt, lab_fgt  = img_ret.to(opt.device), lab_ret.to(opt.device),img_fgt.to(opt.device), lab_fgt.to(opt.device)
                    
                    optimizer.zero_grad()

                    logits_fgt = bbone(img_fgt)

                    # compute pairwise cosine distance between embeddings and distribs
                    dists = self.pairwise_cos_dist(logits_fgt, distribs)


                    if init:
                        closest_distribs = torch.argsort(dists, dim=1)
                        tmp = closest_distribs[:, 0]
                        closest_distribs = torch.where(tmp == lab_fgt, closest_distribs[:, 1], tmp)
                        all_closest_distribs.append(closest_distribs)
                        closest_distribs = all_closest_distribs[-1]
                    else:
                        closest_distribs = all_closest_distribs[n_batch]


                    dists = dists[torch.arange(dists.shape[0]), closest_distribs[:dists.shape[0]]]
                    loss_fgt = torch.mean(dists) * opt.lambda_1

                    logits_ret = bbone(img_ret)
                    outputs_ret = fc(logits_ret)

                    loss_ret = criterion(outputs_ret/opt.temperature, lab_ret)*opt.lambda_2
                    loss = loss_ret+ loss_fgt
                    
                    if n_batch_ret>opt.batch_fgt_ret_ratio:
                        del loss,loss_ret,loss_fgt, logits_fgt, logits_ret, outputs_ret,dists
                        break
                    
                    loss.backward()
                    optimizer.step()

                # evaluate accuracy on forget set every batch
            with torch.no_grad():
                self.net.eval()
                curr_acc = accuracy(self.net, self.forget)
                self.net.train()
                print(f"ACCURACY FORGET SET: {curr_acc:.3f}, target is {opt.target_accuracy:.3f}")
                if curr_acc < opt.target_accuracy:
                    flag_exit = True

            if flag_exit:
                break

            init = False
            scheduler.step()


        self.net.eval()
        return self.net

class Mahalanobis(BaseMethod):
    def __init__(self, net, retain, retain_sur, forget,test,class_to_remove=None):
        super().__init__(net, retain, forget, test)
        self.loader = None
        self.class_to_remove = class_to_remove
        self.retain_sur = retain_sur

    def cov_mat_shrinkage(self,cov_mat,gamma1=opt.gamma1,gamma2=opt.gamma2):
        I = torch.eye(cov_mat.shape[0]).to(opt.device)
        V1 = torch.mean(torch.diagonal(cov_mat))
        off_diag = cov_mat.clone()
        off_diag.fill_diagonal_(0.0)
        mask = off_diag != 0.0
        V2 = (off_diag*mask).sum() / mask.sum()
        cov_mat_shrinked = cov_mat + gamma1*I*V1 + gamma2*(1-I)*V2
        return cov_mat_shrinked
    
    def normalize_cov(self,cov_mat):
        sigma = torch.sqrt(torch.diagonal(cov_mat))  # standard deviations of the variables
        cov_mat = cov_mat/(torch.matmul(sigma.unsqueeze(1),sigma.unsqueeze(0)))
        return cov_mat


    def mahalanobis_dist(self, samples,samples_lab, mean,S_inv):
        #check optimized version
        diff = F.normalize(self.tuckey_transf(samples), p=2, dim=-1)[:,None,:] - F.normalize(mean, p=2, dim=-1)
        right_term = torch.matmul(diff.permute(1,0,2), S_inv)
        mahalanobis = torch.diagonal(torch.matmul(right_term, diff.permute(1,2,0)),dim1=1,dim2=2)
        return mahalanobis

    def distill(self, outputs_ret, outputs_original):

        soft_log_old = torch.nn.functional.log_softmax(outputs_original+10e-5, dim=1)
        soft_log_new = torch.nn.functional.log_softmax(outputs_ret+10e-5, dim=1)

        kl_div = torch.nn.functional.kl_div(soft_log_new+10e-5, soft_log_old+10e-5, reduction='batchmean', log_target=True)

        return kl_div

    def tuckey_transf(self,vectors,beta=opt.beta):
        return torch.pow(vectors,beta)
    
    def pairwise_cos_dist(self, x, y):
        """Compute pairwise cosine distance between two tensors"""
        x_norm = torch.norm(x, dim=1).unsqueeze(1)
        y_norm = torch.norm(y, dim=1).unsqueeze(1)
        x = x / x_norm
        y = y / y_norm
        return 1 - torch.mm(x, y.transpose(0, 1))
 
    def run(self):
        """compute embeddings"""
        #lambda1 fgt
        #lambda2 retain

        original_model = deepcopy(self.net) # self.net
        original_model.eval()
        bbone = torch.nn.Sequential(*(list(self.net.children())[:-1] + [nn.Flatten()]))
        if opt.model == 'AllCNN':
            fc = self.net.classifier
        else:
            fc = self.net.fc
        
        bbone.eval()

 
        # embeddings of retain set
        with torch.no_grad():
            ret_embs=[]
            labs=[]
            cnt=0
            for img_ret, lab_ret in self.retain:
                img_ret, lab_ret = img_ret.to(opt.device), lab_ret.to(opt.device)
                logits_ret = bbone(img_ret)
                ret_embs.append(logits_ret)
                labs.append(lab_ret)
                cnt+=1
            ret_embs=torch.cat(ret_embs)
            labs=torch.cat(labs)
        

        # compute distribs from embeddings
        distribs=[]
        cov_matrix_inv =[]
        for i in range(opt.num_classes):
            if type(self.class_to_remove) is list:
                if i not in self.class_to_remove:
                    samples = self.tuckey_transf(ret_embs[labs==i])
                    distribs.append(samples.mean(0))
                    cov = torch.cov(samples.T)
                    #cov_shrinked = self.cov_mat_shrinkage(cov)
                    cov_shrinked = self.cov_mat_shrinkage(self.cov_mat_shrinkage(cov))
                    cov_shrinked = self.normalize_cov(cov_shrinked)
                    cov_matrix_inv.append(torch.linalg.pinv(cov_shrinked))
            else:
                
                samples = self.tuckey_transf(ret_embs[labs==i])
                distribs.append(samples.mean(0))
                cov = torch.cov(samples.T)
                cov_shrinked = self.cov_mat_shrinkage(self.cov_mat_shrinkage(cov))
                cov_shrinked = self.normalize_cov(cov_shrinked)
                cov_matrix_inv.append(torch.linalg.pinv(cov_shrinked))

        distribs=torch.stack(distribs)
        cov_matrix_inv=torch.stack(cov_matrix_inv)
        
        bbone.train(), fc.train()

        optimizer = optim.Adam(self.net.parameters(), lr=opt.lr_unlearn, weight_decay=opt.wd_unlearn)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.scheduler, gamma=0.5)

        init = True
        flag_exit = False
        all_closest_class = []
        if opt.dataset == 'tinyImagenet':
            ls = 0.2
        else:
            ls = 0
        criterion = nn.CrossEntropyLoss(label_smoothing=ls)
        

        print('Num batch forget: ',len(self.forget), 'Num batch retain: ',len(self.retain_sur))
        print(f'fgt ratio:{opt.batch_fgt_ret_ratio}')
        for _ in tqdm(range(opt.epochs_unlearn)):
            for n_batch, (img_fgt, lab_fgt) in enumerate(self.forget):
                #print('new fgt')
                for n_batch_ret, (img_ret, lab_ret) in enumerate(self.retain_sur):
                    img_ret, lab_ret,img_fgt, lab_fgt  = img_ret.to(opt.device), lab_ret.to(opt.device),img_fgt.to(opt.device), lab_fgt.to(opt.device)
                    optimizer.zero_grad()

                    embs_fgt = bbone(img_fgt)

                    # compute Mahalanobis distance between embeddings and cluster
                    dists = self.mahalanobis_dist(embs_fgt,lab_fgt,distribs,cov_matrix_inv).T
                    #dists = self.pairwise_cos_dist(embs_fgt, distribs)
                    embs_fgt_exp = embs_fgt.unsqueeze(1)
                    distribs_exp = distribs.unsqueeze(0)
                    #dists=torch.nn.functional.kl_div(torch.nn.functional.log_softmax(embs_fgt_exp+1e-6, dim=2), torch.nn.functional.softmax(distribs_exp+1e-6, dim=2), reduction='none').sum(dim=2)
                    #dists=torch.mean((embs_fgt_exp-distribs_exp)**2,dim=2)
                    #dists=torch.norm((embs_fgt_exp-distribs_exp),dim=2)

                    if init:
                        closest_class = torch.argsort(dists, dim=1)
                        tmp = closest_class[:, 0]
                        closest_class = torch.where(tmp == lab_fgt, closest_class[:, 1], tmp)
                        all_closest_class.append(closest_class)
                        closest_class = all_closest_class[-1]
                    else:
                        closest_class = all_closest_class[n_batch]


                    dists = dists[torch.arange(dists.shape[0]), closest_class[:dists.shape[0]]]
                    loss_fgt = torch.mean(dists) * opt.lambda_1

                    outputs_ret = fc(bbone(img_ret))
                    with torch.no_grad():
                        outputs_original = original_model(img_ret)
                        if opt.mode =='CR':
                            label_out = torch.argmax(outputs_original,dim=1)
                            ##### da correggere per multiclass
                            outputs_original = outputs_original[label_out!=self.class_to_remove[0],:]
                            
                            outputs_original[:,torch.tensor(self.class_to_remove,dtype=torch.int64)] = torch.min(outputs_original)
                    if opt.mode =='CR':
                        outputs_ret = outputs_ret[label_out!=self.class_to_remove[0],:]
                    
                    loss_ret = self.distill(outputs_ret, outputs_original)*opt.lambda_2
                    loss=loss_ret+loss_fgt
                    
                    
                    if n_batch_ret>opt.batch_fgt_ret_ratio:
                        del loss,loss_ret,loss_fgt, embs_fgt,dists
                        break
                    print(f'n_batch_ret:{n_batch_ret} ,loss FGT:{loss_fgt}, loss RET:{loss_ret}')
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        self.net.eval()
                        curr_acc = accuracy(self.net, self.forget)
                        #test_acc=accuracy(self.net, self.test)
                        #ret_acc=accuracy(self.net, self.retain)
                        #ret_sur = accuracy(self.net, self.retain_sur)
                        #print(f'Acc fgt set: {curr_acc:.3f} Acc ret set: {ret_acc:.3f}, Acc test: {test_acc:.3f}')
                        self.net.train()
                        if curr_acc < opt.target_accuracy:
                            flag_exit = True

                    if flag_exit:
                        break
                if flag_exit:
                    break

            # evaluate accuracy on forget set every batch
            #this can be removed
            with torch.no_grad():
                self.net.eval()
                curr_acc = accuracy(self.net, self.forget)
                test_acc=accuracy(self.net, self.test)
                self.net.train()
                print(f"AAcc forget: {curr_acc:.3f}, target is {opt.target_accuracy:.3f}, test is {test_acc:.3f}")
                if curr_acc < opt.target_accuracy:
                    flag_exit = True

            if flag_exit:
                break

            init = False
            #scheduler.step()


        self.net.eval()
        return self.net