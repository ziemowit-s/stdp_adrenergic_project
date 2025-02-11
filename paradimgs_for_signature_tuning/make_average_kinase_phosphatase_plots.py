#!/usr/bin/env python
from __future__ import print_function, division, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import glob

colors = {"DA": "gray",
          "ISO": "maroon",
          "LFS": "gold",
          "HFS": "darkgreen",
          "massed": "forestgreen",
           "spaced": "black",
          "ISO+LFS": "orange",
          "DA+LFS": "dodgerblue",
          "ISO+HFS": "midnightblue",
          "t-LTP": "mediumblue",
          "t-LTD": "darkmagenta",}
#          "yellow"]

labels = {
    "DA": "DA",
    "ISO": "ISO",
    "LFS": "LFS",
    "HFS": "HFS",
    "massed": "massed",
    "spaced": "spaced",
    "ISO+LFS": "ISO+LFS",
    "DA+LFS": "DA+LFS",
    "ISO+HFS": "ISO+HFS",
    "t-LTP": "t-LTP",
    "t-LTD": "t-LTD",
}

filename_dict = {    
    "DA": "../cAMP_validation/model_DA_bath_trial",
    "ISO": "../cAMP_validation/model_ISO_bath_trial",
    "LFS": "model_LFS_trial",
    "HFS": "model_HFS_trial",
    "massed": "model_4xHFS_3s_trial",
    "spaced": "model_4xHFS_80s_trial",
    "ISO+LFS": "model_ISO_bath_LFS_trial",
    "DA+LFS": "model_DA_bath_LFS_trial",
    "ISO+HFS": "model_ISO_bath_HFS_trial",
    "t-LTP": "../STDP_paradigms/model_STDP_+10_ms_trial",
    "t-LTD": "../STDP_paradigms/model_STDP_-20_ms_trial",
    }

species = {"CaMKII": [ "CKpCaMCa4", "CKp", #"CKCaMCa4",
                      "GluR1_CKpCaM", "GluR1_CKp",
                      "GluR1_CKpCaM2", "GluR1_CKp2",
                      "GluR1_S845_CKpCaM", "GluR1_S845_CKp",
                      "GluR1_S845_CKpCaM2", "GluR1_S845_CKp2",],
           "Epac":  ["Epac1cAMP"],
           "PKA" : ["GluR1_S845", "GluR1_S845_S831", "GluR1_S845_S567",
                    "Ip35", "Ip35PP1",
                    "pNMDAR",
                    "pPDE4", "pPDE4cAMP",
                    "PKAcpISObAR","PKAcpbAR",
                    "PKAcppISObAR","PKAcppbAR",
                    "PKAcpppISObAR","PKAcpppbAR",
                    "pISObAR", "ppISObAR", "pppISObAR",
                    "ppppISObAR",
                    "pbAR", "ppbAR", "pppbAR", "ppppbAR"],
                    
           "Gi": ["Gibg"],
           "PP1": ["CKpCaMCa4PP1", "CKpPP1", "GluR1_S845_PP1",
                   "GluR1_S567_PP1", "GluR1_S831_PP1",
                   "GluR1_S845_S831_PP1", "GluR1_S845_S567_PP1",
                   "GluR1_S845_S831_PP1_2", "GluR1_S845_S567_PP1_2",
                   "PP1", "PP1pNMDAR"],
           "PP2B":["PP2BCaMCa4", "GluR1_S845_PP2B",
                   "GluR1_S845_S831_PP2B", "GluR1_S845_S567_PP2B",
                   "Ip35PP1PP2BCaMCa4", "Ip35PP2BCaMCa4",
                   "PP2BpNMDAR"],
           "S845": ["GluR1_S845", "GluR1_S845_S831", "GluR1_S845_S567"],
           "S831": ["GluR1_S831", "GluR1_S845_S831"],
           "S567": ["GluR1_S567",  "GluR1_S845_S567"],
           "pNMDAR": ["pNMDAR"],

           }

evaluated_species = ["CaMKII", "PKA", "Epac", "Gi",]# "PP1",
                    # "PP2B"]
receptors = ["S845", "S831", "S567", "pNMDAR"]
endings = ["spine.txt", "dendrite.txt", "PSD.txt"]
basal_specie = {"S845":1200, "S831":1180, "S567":2400, "pNMDAR":800}
def read_in_data(filename, e_species):#basal_fname="../model_start_trial"):
    data_dict = {}
    for j, ending in enumerate(endings):
        pattern = "%s?_%s" % (filename, ending)
        fnames = glob.glob(pattern)
        print(fnames)
        all_data = []
        for z, fname in enumerate(fnames):
            f = open(fname)
            header = f.readline().split()
            data = np.loadtxt(f)
            out = np.zeros((len(data), len(e_species)+1))
            out[:, 0]  = data[:, 0]
            for idx, specie_set in enumerate(e_species):
                this_set = species[specie_set]
                for specie in this_set:
                    k = header.index(specie)
                    out[:, idx+1] += data[:, k]
            all_data.append(out)
            min_len_idx = np.argmin([a_d.shape[0] for a_d in all_data])
            min_len = all_data[min_len_idx].shape[0]
            average_traces = np.zeros((min_len,
                                       len(e_species)+1))
            average_traces[:, 0] = all_data[min_len_idx][:, 0]
            for a_d in all_data:
                average_traces[:, 1:] += a_d[:min_len, 1:]
        data_dict[ending] = average_traces
        for k in range(len(e_species)):
            average_traces[:, k+1] = average_traces[:, k+1]/len(all_data)
    return data_dict
        
        
if __name__ == "__main__":
    keys = [ "HFS", "massed", "spaced", "ISO+HFS", "ISO+LFS",  "t-LTP", "t-LTD",]
    fig1, ax1 = plt.subplots(len(endings)-1,
                             len(evaluated_species),
                             figsize =(5*len(endings),
                                       5*(1+len(evaluated_species))))
    fig2, ax2 = plt.subplots(len(receptors)//2, 2,
                             figsize =(5*len(receptors),
                                       5*2))

    max_values = {specie:[] for specie in evaluated_species}
    min_values = {specie:[] for specie in evaluated_species}
    max_values_r = {specie:[] for specie in receptors}
    min_values_r = {specie:[] for specie in receptors}
    basal = read_in_data("../model_start_trial", evaluated_species+receptors)
    for key in keys:
        datas = read_in_data(filename_dict[key], evaluated_species+receptors)
        for j, ending in enumerate(datas):
            average_traces = datas[ending]
            l_basal = basal[ending].shape[0]
            l = average_traces.shape[0]
            for k, spec in enumerate(evaluated_species+receptors):
                trace = average_traces[:, k+1]
<<<<<<< HEAD
                
                if ending != "PSD.txt":
                    if spec in evaluated_species:
                        if k == 0 and j == 0:
                            ax1[j][k].plot(average_traces[:, 0], trace,
                                           color=colors[key], label=labels[key])
                        else:
                            ax1[j][k].plot(average_traces[:, 0], trace,
                                           color=colors[key])
                else:
                    if spec in receptors:
                        new_k = k - len(evaluated_species)
                        
                        if new_k == 0:
                            ax2[new_k//2][new_k%2].plot(average_traces[:, 0], trace/basal_specie[spec],
=======
                if basal[ending][:, k+1].mean():
                    trace = trace/basal[ending][:, k+1].mean()
                # if evaluated_species[k] == "PP1":
                #     trace = average_traces[:, 1] / trace
                if spec in evaluated_species:
                    if k == 0 and j == 0:
                        ax1[j][k].plot(average_traces[:, 0], trace,
                                       color=colors[key], label=labels[key])
                    else:
                        ax1[j][k].plot(average_traces[:, 0], trace,
                                       color=colors[key])
                elif spec in receptors and ending=="spine.txt":
                    if spec == "S567":
                        trace = trace/average_traces[:, k]
                    new_k = k - len(evaluated_species)
                    if new_k == 0:
                        ax2[new_k//2][new_k%2].plot(average_traces[:, 0], trace,
>>>>>>> zs/master
                                                    color=colors[key], label=labels[key])
                        else:
                            ax2[new_k//2][new_k%2].plot(average_traces[:, 0], trace/basal_specie[spec],
                                                        color=colors[key])

    for j, ending in enumerate(endings[:-1]):
        ax1[j][0].set_ylabel(ending[:-4])
    for idx, specie in enumerate(evaluated_species):
        ax1[0][idx].set_title(specie)
    for idx, specie in enumerate(receptors):
        ax2[idx//2][idx%2].set_title(specie)
    fig1.legend(loc=7)
    fig2.legend(loc=7)
    fig1.savefig("Kinases.png", format="png",
                 bbox_inches='tight', transparent=True, pad_inches=0.1)
    fig2.savefig("Receptor_phosphorylation.png", format="png",
                 bbox_inches='tight', transparent=True, pad_inches=0.1)

    plt.show()
