from copy import deepcopy
from dsets import get_dsets_remove_class, get_dsets
import pandas as pd
from error_propagation import Complex
from utils import accuracy, set_seed, get_retrained_model,get_trained_model
from MIA_code.MIA import get_MIA_SVC
from opts import OPT as opt
import torch.nn as nn
from tqdm import tqdm
import time
from Unlearning_methods import choose_method
from error_propagation import Complex
import os
import torch
if opt.push_results:
    from publisher import push_results

def AUS(a_t, a_or, a_f):
    if opt.mode == "HR":
        aus=(Complex(1, 0)-(a_or-a_t))/(Complex(1, 0)+abs(a_f-a_t))
    else:
        aus=(Complex(1, 0)-(a_or-a_t))/(Complex(1, 0)+abs(a_f))
    return aus

def main(train_fgt_loader, train_retain_loader, seed=0, test_loader=None, test_fgt_loader=None, test_retain_loader=None, class_to_remove=0):
    v_orig, v_unlearn, v_rt = None, None, None
    original_pretr_model = get_trained_model()
    original_pretr_model.to(opt.device)
    original_pretr_model.eval()

    if opt.run_original:
        
        
        if opt.mode =="HR":
            df_or_model = get_MIA_SVC(train_fgt_loader, test_loader, original_pretr_model, opt)
            df_or_model["test_accuracy"] = accuracy(original_pretr_model, test_loader)
        elif opt.mode =="CR":
            df_or_model = pd.DataFrame([0],columns=["PLACEHOLDER"])
            df_or_model["forget_test_accuracy"] = accuracy(original_pretr_model, test_fgt_loader)
            df_or_model["retain_test_accuracy"] = accuracy(original_pretr_model, test_retain_loader)

        df_or_model["forget_accuracy"] = accuracy(original_pretr_model, train_fgt_loader)
        df_or_model["retain_accuracy"] = accuracy(original_pretr_model, train_retain_loader)
        print(df_or_model)
        v_orig= df_or_model.mean(0)
        #convert v_orig back to df
        v_orig = pd.DataFrame(v_orig).T
    #print(df_or_model)

    if opt.run_unlearn:
        print('\n----BEGIN UNLEARNING----')
        pretr_model = deepcopy(original_pretr_model)
        pretr_model.to(opt.device)
        pretr_model.eval()

        timestamp1 = time.time()

        if opt.mode == "HR":
            opt.target_accuracy = accuracy(original_pretr_model, test_loader)

            if opt.method == "DUCK":
                approach = choose_method(opt.method)(pretr_model,train_retain_loader, train_fgt_loader,test_loader, class_to_remove=None)
            else:
                approach = choose_method(opt.method)(pretr_model,train_retain_loader, train_fgt_loader,test_loader)


        elif opt.mode == "CR":
            opt.target_accuracy = 0.01
            if opt.method == "DUCK" or opt.method == "RandomLabels":
                approach = choose_method(opt.method)(pretr_model,train_retain_loader, train_fgt_loader,test_fgt_loader, class_to_remove=class_to_remove)
            else:
                approach = choose_method(opt.method)(pretr_model,train_retain_loader, train_fgt_loader,test_fgt_loader)

        if opt.load_unlearned_model:
            print("LOADING UNLEARNED MODEL")
            if opt.mode == "HR":
                unlearned_model_dict = torch.load(f"{opt.root_folder}/out/{opt.mode}/{opt.dataset}/models/unlearned_model_{opt.method}_seed_{seed}.pth") 
            elif opt.mode == "CR":
                unlearned_model_dict = torch.load(f"{opt.root_folder}/out/{opt.mode}/{opt.dataset}/models/unlearned_model_{opt.method}_seed_{seed}_class_{'_'.join(map(str, class_to_remove))}.pth")

            unlearned_model = get_trained_model().to(opt.device)
            unlearned_model.load_state_dict(unlearned_model_dict)
            print("UNLEARNED MODEL LOADED")
        else:
            unlearned_model = approach.run()

        unlearned_model.eval()
        #save model
        if opt.save_model:
            if opt.mode == "HR":
                torch.save(unlearned_model.state_dict(), f"{opt.root_folder}/out/{opt.mode}/{opt.dataset}/models/unlearned_model_{opt.method}_seed_{seed}.pth")
            elif opt.mode == "CR":
                torch.save(unlearned_model.state_dict(), f"{opt.root_folder}/out/{opt.mode}/{opt.dataset}/models/unlearned_model_{opt.method}_seed_{seed}_class_{'_'.join(map(str, class_to_remove))}.pth")

        unlearn_time = time.time() - timestamp1
        print("BEGIN SVC FIT")
        if opt.mode == "HR":
            df_un_model = get_MIA_SVC(train_fgt_loader, test_loader, unlearned_model, opt)
        elif opt.mode == "CR":
            df_un_model = pd.DataFrame([0],columns=["PLACEHOLDER"])

    
        df_un_model["unlearn_time"] = unlearn_time

        print("UNLEARNING COMPLETED, COMPUTING ACCURACIES...")      
        if opt.mode == "HR":
            df_un_model["test_accuracy"] = accuracy(unlearned_model, test_loader)
        elif opt.mode == "CR":
            df_un_model["forget_test_accuracy"] = accuracy(unlearned_model, test_fgt_loader)
            df_un_model["retain_test_accuracy"] = accuracy(unlearned_model, test_retain_loader)

        df_un_model["forget_accuracy"] = accuracy(unlearned_model, train_fgt_loader)
        df_un_model["retain_accuracy"] = accuracy(unlearned_model, train_retain_loader)
        #print(df_un_model)
        v_unlearn=df_un_model.mean(0)
        v_unlearn = pd.DataFrame(v_unlearn).T
        print("UNLEARN COMPLETED")

    if opt.run_rt_model:
        print('\n----MODEL RETRAINED----')

        rt_model = get_retrained_model()
        rt_model.to(opt.device)
        rt_model.eval()
        if opt.mode == "HR":
            df_rt_model = get_MIA_SVC(train_fgt_loader, test_loader, rt_model, opt)
            df_rt_model["test_accuracy"] = accuracy(rt_model, test_loader)

        elif opt.mode == "CR":
            df_or_model = pd.DataFrame([0],columns=["PLACEHOLDER"])
            df_rt_model["forget_test_accuracy"] = accuracy(rt_model, test_fgt_loader)
            df_rt_model["retain_test_accuracy"] = accuracy(rt_model, test_retain_loader)

        df_rt_model["forget_accuracy"] = accuracy(rt_model, train_fgt_loader)
        df_rt_model["retain_accuracy"] = accuracy(rt_model, train_retain_loader)

        v_rt = df_rt_model.mean(0)
        v_rt = pd.DataFrame(v_rt).T
       
    #save dfs
    if opt.run_unlearn:
        if opt.save_df:
            if opt.mode == "HR":
                v_unlearn.to_csv(f"{opt.root_folder}/out/{opt.mode}/{opt.dataset}/dfs/{opt.method}_seed_{seed}.csv")
            elif opt.mode == "CR":
                v_unlearn.to_csv(f"{opt.root_folder}/out/{opt.mode}/{opt.dataset}/dfs/{opt.method}_seed_{seed}_class_{'_'.join(map(str, class_to_remove))}.csv")
    return v_orig, v_unlearn, v_rt

if __name__ == "__main__":
    df_unlearned_total=[]
    df_retrained_total=[]
    df_orig_total=[]
    
    #create output folders
    if not os.path.exists(opt.root_folder+"out/"+opt.mode+"/"+opt.dataset+"/models"):
        os.makedirs(opt.root_folder+"out/"+opt.mode+"/"+opt.dataset+"/models")
    if not os.path.exists(opt.root_folder+"out/"+opt.mode+"/"+opt.dataset+"/dfs"):
        os.makedirs(opt.root_folder+"out/"+opt.mode+"/"+opt.dataset+"/dfs")

    for i in opt.seed:
        set_seed(i)

        print(f"Seed {i}")
        if opt.mode == "HR":
            if opt.dataset == "cifar10":
                num=5000
            elif opt.dataset == "cifar100":
                num=5000
            elif opt.dataset == "tinyImagenet":
                num=10000
            file_fgt = f'{opt.root_folder}forget_id_files/forget_idx_{num}_{opt.dataset}_seed_{i}.txt'
            train_loader, test_loader, train_fgt_loader, train_retain_loader = get_dsets(file_fgt=file_fgt)
            opt.RT_model_weights_path=opt.root_folder+f'chks_{opt.dataset if opt.dataset!="tinyImagenet" else "tiny"}/chks_{opt.dataset if opt.dataset!="tinyImagenet" else "tiny"}_seed_{i}.pth'
            print(opt.RT_model_weights_path)

            row_orig, row_unl, row_ret=main(train_fgt_loader, train_retain_loader, test_loader=test_loader, seed=i)
            #print all the row_unl dataframe
            print(f"Unlearned: {row_unl}")
            if row_unl is not None:
                df_unlearned_total.append(row_unl)
            if row_orig is not None:
                df_orig_total.append(row_orig)
            if row_ret is not None:
                df_retrained_total.append(row_ret)

        elif opt.mode == "CR":
            for class_to_remove in opt.class_to_remove:
                print(f'------------class {class_to_remove}-----------')
                _, _, train_fgt_loader, train_retain_loader, test_fgt_loader, test_retain_loader = get_dsets_remove_class(class_to_remove)

                opt.RT_model_weights_path = opt.root_folder+f'weights/chks_{opt.dataset if opt.dataset!="tinyImagenet" else "tiny"}/best_checkpoint_without_{class_to_remove}.pth'
                print(opt.RT_model_weights_path)

                row_orig, row_unl, row_ret=main(train_fgt_loader, train_retain_loader, test_fgt_loader=test_fgt_loader, seed=i, test_retain_loader=test_retain_loader, class_to_remove=class_to_remove)

                #print results
                

                
                if row_orig is not None:
                    print(f"Original retain test acc: {row_orig['retain_test_accuracy']}")
                    df_orig_total.append(row_orig)
                if row_unl is not None:
                    print(f"Unlearned retain test acc: {row_unl['retain_test_accuracy']}")
                    df_unlearned_total.append(row_unl)
                if row_ret is not None:
                    print(f"Retrained retain test acc: {row_ret['retain_test_accuracy']}")
                    df_retrained_total.append(row_ret)
        
    print(opt.dataset)
    #create results folder if doesn't exist
    
    dfs = {"orig":[], "unlearned":[], "retrained":[]}
    for name, df in zip(dfs.keys(),[df_orig_total, df_unlearned_total, df_retrained_total]):
        if df:
            print("{name} \n")
            #merge list of pd dataframes
            dfs[name] = pd.concat(df)

            means = dfs[name].mean()
            std_devs = dfs[name].std()
            output = "\n".join([f"{col}: {100*mean:.2f} \\pm {100*std:.2f}" if col != 'unlearning_time' else f"{col}: {mean:.2f} \\pm {std:.2f}" for col, mean, std in zip(means.index, means, std_devs)])
            print(output)

            if opt.mode == "HR":
                a_t = Complex(means["test_accuracy"], std_devs["test_accuracy"])
                a_f = Complex(means["forget_accuracy"], std_devs["forget_accuracy"])
                a_or = opt.a_or[opt.dataset][0]
                deltaF1 = abs(dfs[name]["F1"]*100-50).mean()
                deltaF1_std = abs(dfs[name]["F1"]*100-50).std()
                print(f"deltaF1: {deltaF1:.4f} \pm {deltaF1_std:.4f}")

            elif opt.mode == "CR":
                a_t = Complex(means["retain_test_accuracy"], std_devs["retain_test_accuracy"])
                a_f = Complex(means["forget_test_accuracy"], std_devs["forget_test_accuracy"])
                a_or = opt.a_or[opt.dataset][1]
            aus = AUS(a_t, a_or, a_f)
            dfs[name]["AUS"] = aus.value
            print(f"AUS: {aus.value:.4f} \pm {aus.error:.4f}")
   

    if opt.push_results:
        push_results(opt, dfs["orig"], dfs["unlearned"], dfs["retrained"])


