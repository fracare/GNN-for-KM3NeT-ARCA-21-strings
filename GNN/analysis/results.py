import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py

inputfile = str(sys.argv[1])

tag = str(sys.argv[2])
if(tag=='bg'):
    ind = 0
elif(tag=='ts'):
    ind = 1
else:
    exit()

def histo(x,y):

    N = 100000
    x_histo = np.linspace(0,1,N)
    y_histo = np.zeros(N)

    for i in range(N):
        for j in range(len(x)-1):
            if(x_histo[i] > x[j] and x_histo[i] < x[j+1]):
                y_histo[i] = y[j]
                break

    return x_histo, y_histo

# load score data
with h5py.File(inputfile, 'r') as f:
    true   = f["label_"+tag+"_output"][()]    
    pred   = f["pred_"+tag+"_output"][()]
    ntrig  = f["y_values"]["n_trig_hits"]
    tr_lik = f["y_values"]["jmuon_likelihood"]
    pid    = f["y_values"]["particle_type"]
    is_cc  = f["y_values"]["is_cc"]
    
    name = 'MC muons & neutrinos'
    name = r'MC muons & $\nu_{e}$CC'
   
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

    muon_score = []
    neutrino_score = []

    for j in range (len(cut)):

        TP = 0 # True Positive
        TN = 0 # True Negative
        FP = 0 # False Positive
        FN = 0 # False Negative

        for i in range(len(true)):
        
            if (True):
            #if ((np.abs(pid[i])==12 and is_cc[i]==2) or (np.abs(pid[i])==13)):
            #if (np.abs(pid[i])!=16):
            #if (ntrig[i] >= 20 and not np.isnan(tr_lik[i]) and tr_lik[i]>1e-5):

                if (float(pred[i][ind]) > cut[j] and float(true[i][ind]) == 1):
    
                    TP += 1
                    if(j==0):
                        neutrino_score.append(float(pred[i][ind]))
        
                elif (float(pred[i][ind]) > cut[j] and float(true[i][ind]) == 0):
     
                    FP += 1
                    if(j==0):
                        muon_score.append(float(pred[i][ind]))
        
                elif (float(pred[i][ind]) < cut[j] and float(true[i][ind]) == 0):
     
                    TN += 1
                    if(j==0):
                        muon_score.append(float(pred[i][ind]))
    
                elif (float(pred[i][ind]) < cut[j] and float(true[i][ind]) == 1):
     
                    FN += 1
                    if(j==0):
                        neutrino_score.append(float(pred[i][ind]))

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


cut_histo, accuracy_histo = histo(cut, accuracy)
cut_histo, FN_rate_histo = histo(cut, FN_rate)
cut_histo, FP_rate_histo = histo(cut, FP_rate)
cut_histo, recall_histo = histo(cut, recall)
cut_histo, specificity_histo = histo(cut, specificity)
cut_histo, precision_histo = histo(cut, precision)
cut_histo, false_alarm_histo = histo(cut, false_alarm)

plt.rcParams.update({'font.size': 18})


if(tag == 'bg'):
    class0 = "atm muons"
    class1 = "neutrinos"
elif(tag == 'ts'):
    class0 = "tracks"
    class1 = "showers"

fig=plt.figure(figsize=(8,7))
plt.hist(neutrino_score, bins = 100, range=(0,1), density=False, log=True, label=class1, histtype='step',linewidth=2)
plt.hist(muon_score, bins = 100, range=(0,1), density=False, log=True, label=class0, histtype='step',linewidth=2)
plt.axvline(cut[Gmean_ind], color="r", linewidth = 1.5, label='best cut = '+str(round(cut[Gmean_ind],3)), linestyle='--')

if(tag == 'bg'):
    plt.xlabel("neutrino score", fontsize=20)
elif(tag == 'ts'):
    plt.xlabel("shower score", fontsize=20)

plt.ylabel("# events", fontsize=20)
plt.legend(fontsize=20)
plt.legend(loc='upper center', fontsize=20)
#plt.show()
plt.savefig("showerscore.png")


fig=plt.figure(figsize=(8,7))
plt.plot([0,1], [0,1], linestyle='--', label='No skill', zorder=0) 
plt.plot(FP_rate, recall, label='ROC curve', zorder=1)
plt.scatter(FP_rate, recall, color='orange', zorder=2)
plt.plot([0,0,1], [0,1,1], linestyle='--', label='Perfect model', zorder=0)
plt.scatter(FP_rate[Gmean_ind], recall[Gmean_ind], label='Best cut = '+str(round(cut[Gmean_ind],3)), color='black', zorder=3)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Optimal threshold for ROC curve")
plt.legend()
#plt.show()
plt.savefig("ROCcurve.png")


fig, ax1 = plt.subplots(1,1,figsize=(9,6))

ax2 = ax1.twinx()
ax1.axvline(cut[Gmean_ind], color="black", linewidth = 1.5, label='ROC cut = '+str(round(cut[Gmean_ind],3)), linestyle='--', zorder=1)
ax1.plot(cut[:-1], np.array(recall[:-1])*100, c='g', zorder=1)
ax2.plot(cut[:-1], np.array(FP_rate[:-1])*100, c='r', zorder=1)
ax1.scatter(cut[:-1], np.array(recall[:-1])*100, c='g')
ax1.scatter(cut[Gmean_ind], recall[Gmean_ind]*100, c='orange', label='TPR = '+str(round(recall[Gmean_ind]*100,1)) + '%', s=100, zorder=2)
ax2.scatter(cut[:-1], np.array(FP_rate[:-1])*100, c='r')
ax2.scatter(cut[Gmean_ind], FP_rate[Gmean_ind]*100, c='b', label='FPR = '+str(round(FP_rate[Gmean_ind]*100,1)) + '%', s=100, zorder=2)

ax1.set_xlabel('shower score cut', fontsize=18)
ax1.set_ylabel('True Positive Rate (%)', color='g', fontsize=18)
ax2.set_ylabel('False Positive Rate (%)', color='r', fontsize=18)


fig.legend(fontsize=15)
#plt.show()
plt.savefig("TPRFPRvsSCORE.png")

fig, ax1 = plt.subplots(1,1,figsize=(9,6))

ax2 = ax1.twinx()
ax1.axvline(cut[-11], color="black", linewidth = 1.5, label='cut = 0.9', linestyle='--', zorder=1)
ax1.plot(cut[:-1], np.array(recall[:-1])*100, c='g', zorder=1)
ax2.plot(cut[:-1], np.array(FP_rate[:-1])*100, c='r', zorder=1)
ax1.scatter(cut[:-1], np.array(recall[:-1])*100, c='g')
ax1.scatter(cut[-11], recall[-11]*100, c='orange', label='TPR = '+str(round(recall[-11]*100,1)) + '%', s=100, zorder=2)
ax2.scatter(cut[:-1], np.array(FP_rate[:-1])*100, c='r')
ax2.scatter(cut[-11], FP_rate[-11]*100, c='b', label='FPR = '+str(round(FP_rate[-11]*100,2)) + '%', s=100, zorder=2)

ax1.set_xlabel('shower score cut', fontsize=18)
ax1.set_ylabel('True Positive Rate (%)', color='g', fontsize=18)
ax2.set_ylabel('False Positive Rate (%)', color='r', fontsize=18)



fig.legend(fontsize=15)
#plt.show()
plt.savefig("TPRFPRvsSCORE_cutat09.png")


fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.axvline(cut[-2], color="black", linewidth = 1.5, label='cut = '+str(round(cut[-2],3)), linestyle='--', zorder=1)
ax1.plot(cut[:-1], np.array(recall[:-1])*100, c='g', zorder=1)
ax2.plot(cut[:-1], np.array(FP_rate[:-1])*100, c='r', zorder=1)
ax1.scatter(cut[:-1], np.array(recall[:-1])*100, c='g')
ax1.scatter(cut[-2], recall[-2]*100, c='orange', label='TPR = '+str(round(recall[-2]*100,1)) + '%', s=100, zorder=2)
ax2.scatter(cut[:-1], np.array(FP_rate[:-1])*100, c='r')
ax2.scatter(cut[-2], FP_rate[-2]*100, c='b', label='FPR = '+str(round(FP_rate[-2]*100,2)) + '%', s=100, zorder=2)

ax1.set_xlabel('shower score cut', fontsize=18)
ax1.set_ylabel('True Positive Rate (%)', color='g', fontsize=18)
ax2.set_ylabel('False Positive Rate (%)', color='r', fontsize=18)


fig.legend(fontsize=15)
#plt.show()
plt.savefig("TPRFPRvsSCORE_cutat"+str(round(cut[-2],3))+".png")






















