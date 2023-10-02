import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py

inputfile = str(sys.argv[1])
tag       = str(sys.argv[2])
particle  = str(sys.argv[3])
outpath   = str(sys.argv[4])

if(tag=='bg'):
    ind = 0
    if(particle=='all'):
        name = 'MC muons & neutrinos'
    elif(particle=='nueCC'):
        name = r'MC muons & $\overset{(-)}{\nu}_{e}$CC'
    elif(particle=='nueNC'):
        name = r'MC muons & $\overset{(-)}{\nu}_{e}$NC'
    elif(particle=='numuCC'):
        name = r'MC muons & $\overset{(-)}{\nu}_{\mu}$CC'
    elif(particle=='numuNC'):
        name = r'MC muons & $\overset{(-)}{\nu}_{\mu}$NC'
elif(tag=='ts'):
    ind = 0
    if(particle=='muon'):
        name = 'MC muons & showers'
    elif(particle=='numuCC'):
        name = r'MC $\overset{(-)}{\nu}_{\mu}$CC & showers'
else:
    exit()

# load score data
with h5py.File(inputfile, 'r') as f:
    true   = f["label_"+tag+"_output"][()]    
    pred   = f["pred_"+tag+"_output"][()]
    ntrig  = f["y_values"]["n_trig_hits"]
    tr_lik = f["y_values"]["jmuon_likelihood"]
    pid    = f["y_values"]["particle_type"]
    is_cc  = f["y_values"]["is_cc"]
   
    cut = np.linspace(0,1,100)

    recall = [] #  from all positive classes, how many we predicted correctly
    specificity = []
    FP_rate = []
    FN_rate = []
    precision = [] # from all the classes we have predicted as positive, how many are actually positive
    Fmeasure = [] # harmonic mean of Recall and Precision
    Gmean = []
    accuracy = [] # from all the classes, how many of them we have predicted correctly
    false_alarm = []

    class0_score = []
    class1_score = []

    for j in range (len(cut)):

        TP = 0 # True Positive
        TN = 0 # True Negative
        FP = 0 # False Positive
        FN = 0 # False Negative

        for i in range(len(true)):
        
            if(tag=='bg'):
                if(particle=='all'):
                    condition = np.abs(pid[i])!=16
                elif(particle=='nueCC'):
                    condition1 = np.logical_and(np.abs(pid[i])==12, is_cc[i]==2)
                    condition  = np.logical_or(condition1, np.abs(pid[i])==13)
                elif(particle=='nueNC'):
                    condition1 = np.logical_and(np.abs(pid[i])==12, is_cc[i]!=2)
                    condition  = np.logical_or(condition1, np.abs(pid[i])==13)
                elif(particle=='numuCC'):
                    condition1 = np.logical_and(np.abs(pid[i])==14, is_cc[i]==2)
                    condition  = np.logical_or(condition1, np.abs(pid[i])==13)
                elif(particle=='numuNC'):
                    condition1 = np.logical_and(np.abs(pid[i])==14, is_cc[i]!=2)
                    condition  = np.logical_or(condition1, np.abs(pid[i])==13)
            elif(tag=='ts'):
                if(particle=='muon'):
                    condition1 = np.logical_or(np.abs(pid[i])==12, np.abs(pid[i])==13)
                    condition2 = np.logical_and(np.abs(pid[i])==14, is_cc[i]!=2)
                    condition  = np.logical_or(condition1, condition2)
                elif(particle=='numuCC'):
                    condition = np.logical_and(np.abs(pid[i])!=13, np.abs(pid[i])!=16)
        
            if (condition):

                if (float(pred[i][ind]) > cut[j] and float(true[i][ind]) == 1):
                    TP += 1
                    if(j==0):
                        class1_score.append(float(pred[i][ind]))
                elif (float(pred[i][ind]) > cut[j] and float(true[i][ind]) == 0):
                    FP += 1
                    if(j==0):
                        class0_score.append(float(pred[i][ind]))
                elif (float(pred[i][ind]) < cut[j] and float(true[i][ind]) == 0):
                    TN += 1
                    if(j==0):
                        class0_score.append(float(pred[i][ind]))
                elif (float(pred[i][ind]) < cut[j] and float(true[i][ind]) == 1):
                    FN += 1
                    if(j==0):
                        class1_score.append(float(pred[i][ind]))


        recall.append(TP/(TP+FN))
        specificity.append(TN/(FP+TN))
        FP_rate.append(1 - TN/(FP+TN))
        FN_rate.append(FN/(TP+FN))
        if(TP == 0):
            precision.append(1)
        else:
            precision.append(TP/(TP+FP))
        if(FP == 0):
            false_alarm.append(0)
        else:
            false_alarm.append(FP/(FP+TP))
        Fmeasure.append(2.*recall[-1]*precision[-1]/(recall[-1] + precision[-1]))
        Gmean.append(np.sqrt(recall[-1]*specificity[-1]))

        accuracy.append((TP+TN)/(TP+TN+FP+FN))

Gmean_max = max(Gmean)
Gmean_ind = Gmean.index(Gmean_max)

Fmeasure_max = max(Fmeasure)
Fmeasure_ind = Fmeasure.index(Fmeasure_max)

accuracy_max = max(accuracy)
accuracy_ind = accuracy.index(accuracy_max)

false_alarm_max = max(false_alarm)
false_alarm_ind = false_alarm.index(false_alarm_max)

plt.rcParams.update({'font.size': 18})

if(tag == 'bg'):
    label0 = "atm muons"
    label1 = "neutrinos"
elif(tag == 'ts'):
    label0 = "showers"
    label1 = "tracks"

fig=plt.figure(figsize=(8,7))
plt.hist(class0_score, bins = 100, range=(0,1), density=False, log=True, label=label0, histtype='step',linewidth=2)
plt.hist(class1_score, bins = 100, range=(0,1), density=False, log=True, label=label1, histtype='step',linewidth=2)
plt.axvline(cut[Gmean_ind], color="r", linewidth = 1.5, label='best cut = '+str(round(cut[Gmean_ind],3)), linestyle='--')
plt.title(name, fontsize=20)
if(tag == 'bg'):
    plt.xlabel("neutrino score", fontsize=18)
elif(tag == 'ts'):
    plt.xlabel("track score", fontsize=18)
plt.ylabel("# events", fontsize=18)
plt.legend(loc='upper center', fontsize=18)
plt.savefig(outpath+'slide_'+particle+'_score.png')
#plt.show()



fig=plt.figure(figsize=(8,7))
plt.plot([0,1], [0,1], linestyle='--', label='No skill', zorder=0) 
plt.plot(FP_rate, recall, label='ROC curve', zorder=1)
plt.scatter(FP_rate, recall, color='orange', zorder=2)
plt.plot([0,0,1], [0,1,1], linestyle='--', label='Perfect model', zorder=0)
plt.scatter(FP_rate[Gmean_ind], recall[Gmean_ind], label='Best cut = '+str(round(cut[Gmean_ind],3)), color='black', zorder=3)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(name, fontsize=20)
plt.legend()
plt.savefig(outpath+'slide_'+particle+'_roc.png')
#plt.show()


fig, ax1 = plt.subplots(1,1,figsize=(9,6))
ax2 = ax1.twinx()
ax1.axvline(cut[Gmean_ind], color="black", linewidth = 1.5, label='ROC cut = '+str(round(cut[Gmean_ind],3)), linestyle='--', zorder=1)
ax1.plot(cut[:-1], np.array(recall[:-1])*100, c='g', zorder=1)
ax2.plot(cut[:-1], np.array(FP_rate[:-1])*100, c='r', zorder=1)
ax1.scatter(cut[:-1], np.array(recall[:-1])*100, c='g')
ax1.scatter(cut[Gmean_ind], recall[Gmean_ind]*100, c='orange', label='TPR = '+str(round(recall[Gmean_ind]*100,1)) + '%', s=100, zorder=2)
ax2.scatter(cut[:-1], np.array(FP_rate[:-1])*100, c='r')
ax2.scatter(cut[Gmean_ind], FP_rate[Gmean_ind]*100, c='b', label='FPR = '+str(round(FP_rate[Gmean_ind]*100,1)) + '%', s=100, zorder=2)
if(tag == 'bg'):
    ax1.set_xlabel('neutrino score cut', fontsize=18)
elif(tag == 'ts'):
    ax1.set_xlabel('track score cut', fontsize=18)
ax1.set_ylabel('True Positive Rate (%)', color='g', fontsize=18)
ax2.set_ylabel('False Positive Rate (%)', color='r', fontsize=18)
plt.title(name, fontsize=20)
#plt.title(name+' with track lik > 0 & NTrigHits >= 20', fontsize=20)
fig.legend(fontsize=15)
plt.savefig(outpath+'slide_'+particle+'_cut_roc.png')
#plt.show()

fig, ax1 = plt.subplots(1,1,figsize=(9,6))
ax2 = ax1.twinx()
ax1.axvline(cut[-11], color="black", linewidth = 1.5, label='cut = 0.9', linestyle='--', zorder=1)
ax1.plot(cut[:-1], np.array(recall[:-1])*100, c='g', zorder=1)
ax2.plot(cut[:-1], np.array(FP_rate[:-1])*100, c='r', zorder=1)
ax1.scatter(cut[:-1], np.array(recall[:-1])*100, c='g')
ax1.scatter(cut[-11], recall[-11]*100, c='orange', label='TPR = '+str(round(recall[-11]*100,1)) + '%', s=100, zorder=2)
ax2.scatter(cut[:-1], np.array(FP_rate[:-1])*100, c='r')
ax2.scatter(cut[-11], FP_rate[-11]*100, c='b', label='FPR = '+str(round(FP_rate[-11]*100,2)) + '%', s=100, zorder=2)
if(tag == 'bg'):
    ax1.set_xlabel('neutrino score cut', fontsize=18)
elif(tag == 'ts'):
    ax1.set_xlabel('track score cut', fontsize=18)
ax1.set_ylabel('True Positive Rate (%)', color='g', fontsize=18)
ax2.set_ylabel('False Positive Rate (%)', color='r', fontsize=18)
plt.title(name, fontsize=20)
#plt.title(name+' with track lik > 0 & NTrigHits >= 20', fontsize=20)
fig.legend(fontsize=15)
plt.savefig(outpath+'slide_'+particle+'_cut_9.png')
#plt.show()

fig, ax1 = plt.subplots(1,1,figsize=(9,6))
ax2 = ax1.twinx()
ax1.axvline(cut[-2], color="black", linewidth = 1.5, label='cut = '+str(round(cut[-2],3)), linestyle='--', zorder=1)
ax1.plot(cut[:-1], np.array(recall[:-1])*100, c='g', zorder=1)
ax2.plot(cut[:-1], np.array(FP_rate[:-1])*100, c='r', zorder=1)
ax1.scatter(cut[:-1], np.array(recall[:-1])*100, c='g')
ax1.scatter(cut[-2], recall[-2]*100, c='orange', label='TPR = '+str(round(recall[-2]*100,1)) + '%', s=100, zorder=2)
ax2.scatter(cut[:-1], np.array(FP_rate[:-1])*100, c='r')
ax2.scatter(cut[-2], FP_rate[-2]*100, c='b', label='FPR = '+str(round(FP_rate[-2]*100,2)) + '%', s=100, zorder=2)
if(tag == 'bg'):
    ax1.set_xlabel('neutrino score cut', fontsize=18)
elif(tag == 'ts'):
    ax1.set_xlabel('track score cut', fontsize=18)
ax1.set_ylabel('True Positive Rate (%)', color='g', fontsize=18)
ax2.set_ylabel('False Positive Rate (%)', color='r', fontsize=18)
plt.title(name, fontsize=20)
#plt.title(name+' with track lik > 0 & NTrigHits >= 20', fontsize=20)
fig.legend(fontsize=15)
plt.savefig(outpath+'slide_'+particle+'_cut_99.png')
#plt.show()


