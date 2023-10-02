import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
from matplotlib.colors import LogNorm
import tqdm

inputfile = str(sys.argv[1])
tag = str(sys.argv[2])

if (tag=='bg'):
    id = 0
elif (tag=='ts'):
    id = 1

def score(inputfile):
    with h5py.File(inputfile, "r") as f:
        return f['pred_'+tag+'_output'][()]
        
def label(inputfile):
    with h5py.File(inputfile, "r") as f:
        return f['label_'+tag+'_output'][()]

def read(inputfile, variable):
    with h5py.File(inputfile, "r") as f:
        return f['y_values'][variable]

score_cut = 0.99
ntrig_cut = 15

pred  = score(inputfile)
label = label(inputfile)

ntrig     = read(inputfile,'n_trig_hits')
nsnap     = read(inputfile,'n_hits')
pid       = read(inputfile,'particle_type')
cc        = read(inputfile,'is_cc')
tau       = read(inputfile,'tau_topology')
dirx      = read(inputfile,'dir_x')
diry      = read(inputfile,'dir_y')
dirz      = read(inputfile,'dir_z')
sh_dirz   = read(inputfile,'aashower_dir_z')
tr_dirz   = read(inputfile,'jmuon_dir_z')
sh_len    = read(inputfile,'aashower_length')
tr_len    = read(inputfile,'jmuon_length')
sh_lik    = read(inputfile,'aashower_likelihood')
tr_lik    = read(inputfile,'jmuon_likelihood')
energy    = read(inputfile,'energy')

score = []
selected = []
count = 0

energy_histo = []
pid_histo = []
dirz_histo = []
ntrig_histo = []
nsnap_histo = []

theta = []

tr_len_histo  = []
tr_lik_histo  = []
sh_len_histo  = []
sh_lik_histo  = []
score_histo   = []
tr_score_histo = []
sh_score_histo = []
tr_dirz_histo = []
sh_dirz_histo = []

name = r'$\nu_{e}$CC'
#name = r'$\nu_{\mu}$NC'
#name = 'muons'
#name = 'tracks & showers'

for i in range (len(ntrig)):

    #if(True):
    #if (label[i][id] == 0 and ntrig[i]>=0 and np.abs(pid[i])!=16):
    #if (np.abs(pid[i])==13 and ntrig[i]>=0):
    #if (np.abs(pid[i])!=16 and ntrig[i]>=0):
    if (np.abs(pid[i])==12 and cc[i]==2 and ntrig[i]>=0):
    
        score.append(pred[i][id])
        
        dirz_histo.append(dirz[i])
        #ntrig_histo.append(ntrig[i])
        #energy_histo.append(energy[i])

        if (score[-1] > score_cut):
            count += 1
            selected.append(score[-1]) 

        #if (not np.isnan(tr_lik[i]) and not np.isnan(sh_lik[i]) and tr_lik[i]>1e-5 and sh_lik[i]<0):
        if (not np.isnan(tr_lik[i]) and tr_lik[i]>1e-5):
            tr_lik_histo.append(tr_lik[i])
            tr_dirz_histo.append(tr_dirz[i])
            tr_score_histo.append(pred[i][id])
        if (not np.isnan(sh_lik[i]) and sh_lik[i]<0):
            sh_lik_histo.append(sh_lik[i])
            sh_dirz_histo.append(sh_dirz[i])
            sh_score_histo.append(pred[i][id])
            
        if (energy[i] > 40):
            energy_histo.append(energy[i])   
            ntrig_histo.append(ntrig[i])
            score_histo.append(pred[i][id])      
            
            
plt.rcParams.update({'font.size': 18})


hist, bins = np.histogram(ntrig_histo, bins=50)
logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))

plt.hist(ntrig_histo, bins=logbins, log=True, label='entries = '+str(len(ntrig_histo)))
plt.axvline(np.mean(ntrig_histo), label='Average: ' + str(round(np.mean(ntrig_histo),1)), c='black', linewidth = 2)
plt.axvline(np.median(ntrig_histo), label='Median: ' + str(round(np.median(ntrig_histo),1)), c='darkorange', linewidth=2)
plt.axvline(20, linewidth=3.0, c='r', label='Noise cut', linestyle='--')
plt.title('MC '+name, fontsize=20)
plt.xlabel('# triggered hits', fontsize=20)
plt.ylabel('# events', fontsize=20)
plt.legend()
plt.xscale('log')
plt.show()

bins2 = np.linspace(0,1,len(bins))

plt.hist2d(ntrig_histo, score_histo, bins=[logbins,bins2], norm=LogNorm())
plt.colorbar().set_label(label='# events', size=18)
plt.axvline(20, linewidth=3.0, c='r', label='Noise cut')
plt.title('MC '+name, fontsize=20)
plt.xlabel('# triggered hits', fontsize=20)
plt.ylabel('neutrino score', fontsize=20)
plt.xscale('log')
plt.show()



plt.hist(-1*np.array(dirz_histo), bins=50, range=(-1,1), log=True, label='entries = '+str(len(dirz_histo)))
plt.title('MC '+name, fontsize=20)
plt.xlabel('cos(zen)', fontsize=20)
plt.ylabel('# events', fontsize=20)
plt.ylim(bottom=10)
plt.legend()
plt.show()

plt.hist(-1*np.array(tr_dirz_histo), bins=50, range=(-1,1), log=True, label='entries = '+str(len(tr_dirz_histo)))
plt.title('MC '+name+' with track lik > 0', fontsize=20)
plt.xlabel('track reco cos(zen)', fontsize=20)
plt.ylabel('# events', fontsize=20)
plt.ylim(bottom=10)
plt.legend()
plt.show()

plt.hist(-1*np.array(sh_dirz_histo), bins=50, range=(-1,1), log=True, label='entries = '+str(len(sh_dirz_histo)))
plt.title('MC '+name+' with shower lik < 0', fontsize=20)
plt.xlabel('shower reco cos(zen)', fontsize=20)
plt.ylabel('# events', fontsize=20)
plt.ylim(bottom=10)
plt.legend()
plt.show()


plt.hist(score, bins=50, log=True, label='entries = '+str(len(score)))
plt.title('MC '+name, fontsize=20)
plt.xlabel('shower score', fontsize=20)
plt.ylabel('# events', fontsize=20)
plt.legend()
plt.show()


plt.hist2d(-1*np.array(dirz_histo), score, bins=70, range=[(-1,1),(0,1)], norm=LogNorm())
plt.xlim(-1.,1.)
plt.colorbar().set_label(label='# events', size=18)
plt.title('MC '+name, fontsize=20)
plt.xlabel('cos(zen)', fontsize=20)
plt.ylabel('neutrino score', fontsize=20)
plt.show()

plt.hist2d(-1*np.array(tr_dirz_histo), tr_score_histo, bins=70, range=[(-1,1),(0,1)], norm=LogNorm())
plt.xlim(-1.,1.)
plt.colorbar().set_label(label='# events', size=18)
plt.title('MC '+name+' with track lik > 0', fontsize=20)
#plt.title('MC '+name+' with track lik > 0 & NTrigHits >= 20', fontsize=20)
plt.xlabel('track reco cos(zen)', fontsize=20)
plt.ylabel('neutrino score', fontsize=20)
plt.show()

plt.hist2d(-1*np.array(sh_dirz_histo), sh_score_histo, bins=70, range=[(-1,1),(0,1)], norm=LogNorm())
plt.xlim(-1.,1.)
plt.colorbar().set_label(label='# events', size=18)
plt.title('MC '+name+' with shower lik < 0', fontsize=20)
plt.xlabel('shower reco cos(zen)', fontsize=20)
plt.ylabel('neutrino score', fontsize=20)
plt.show()



hist, bins = np.histogram(energy_histo, bins=70)
logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
bins2 = np.linspace(0,1,70)

plt.hist(energy_histo, bins=logbins, log=True, label='entries = '+str(len(energy_histo)))
plt.title('MC '+name, fontsize=20)
plt.xlabel('energy [GeV]', fontsize=20)
plt.ylabel('# events', fontsize=20)
plt.legend()
plt.xscale('log')
plt.show()

plt.hist2d(energy_histo, score_histo, bins=[logbins,bins2], norm=LogNorm())
plt.colorbar().set_label(label='# events', size=18)
plt.title('MC '+name, fontsize=20)
plt.xlabel('energy [GeV]', fontsize=20)
plt.ylabel('neutrino score', fontsize=20)
plt.xscale('log')
plt.show()


'''
plt.hist(tr_lik_histo, bins=50, log=True, label='entries = '+str(len(tr_lik_histo)))
plt.axvline(np.mean(tr_lik_histo), label='Average: ' + str(round(np.mean(tr_lik_histo),1)), c='black', linewidth = 2)
plt.axvline(np.median(tr_lik_histo), label='Median: ' + str(round(np.median(tr_lik_histo),1)), c='r', linewidth = 2)
plt.title('MC '+name, fontsize=20)
plt.xlabel('track reco likelihood', fontsize=20)
plt.ylabel('# events', fontsize=20)
plt.legend()
plt.show()

plt.hist(-1*np.array(sh_lik_histo), bins=50, log=True, label='entries = '+str(len(sh_lik_histo)))
plt.axvline(-1*np.mean(sh_lik_histo), label='Average: ' + str(-1*round(np.mean(sh_lik_histo),1)), c='black', linewidth = 2)
plt.axvline(-1*np.median(sh_lik_histo), label='Median: ' + str(-1*round(np.median(sh_lik_histo),1)), c='r', linewidth = 2)
plt.title('MC '+name, fontsize=20)
plt.xlabel('shower reco likelihood', fontsize=20)
plt.ylabel('# events', fontsize=20)
plt.legend()
plt.show()
'''

'''
plt.hist2d(tr_lik_histo, tr_score_histo, bins=70, norm=LogNorm())
plt.title('MC '+name+' with track lik > 0', fontsize=20)
plt.xlabel('track reco likelihood')
plt.ylabel('shower score')
plt.colorbar().set_label(label='# events', size=18)
plt.show()

plt.hist2d(-1*np.array(sh_lik_histo), sh_score_histo, bins=70, norm=LogNorm())
plt.axvline(np.mean(np.log10(sh_lik_histo)), label='Average: ' + str(-1*round(10**np.mean(np.log10(sh_lik_histo)),1)), c='black', linewidth = 2)
plt.axvline(np.median(np.log10(sh_lik_histo)), label='Median: ' + str(-1*round(10**np.median(np.log10(sh_lik_histo)),1)), c='r', linewidth = 2)
plt.xlabel('shower reco likelihood')
plt.ylabel('shower score')
plt.title('MC '+name+' with shower lik < 0', fontsize=20)
plt.colorbar().set_label(label='# events', size=18)
plt.show()
'''

'''
logbins_tr = np.logspace(-3,np.log10(25000),71)
logbins_sh = np.logspace(0,5,71)
bins_tr = np.linspace(0,4000,71)
bins_sh = np.linspace(-25000,0,71)
#bins_tr = np.linspace(0,1000,71)
#bins_sh = np.linspace(-5000,0,71)
#plt.hist2d(tr_lik_histo, sh_lik_histo, bins=[logbins_tr,logbins_sh], norm=LogNorm())
plt.hist2d(tr_lik_histo, sh_lik_histo, bins=[bins_tr,bins_sh], norm=LogNorm())
plt.ylabel('shower reco likelihood', fontsize=20)
plt.xlabel('track reco likelihood', fontsize=20)
plt.colorbar().set_label(label='# events', size=18)
plt.title('MC '+name+' with track lik > 0 & shower lik < 0', fontsize=20)
#plt.title('MC tracks & NTrigHits >= 20')
#plt.xscale('log')
#plt.yscale('log')
plt.show()



bins = 70
xbins = np.linspace(0,4000,bins+1)
ybins = np.linspace(-25000,0,bins+1)
#xbins = np.linspace(0,1000,bins+1)
#ybins = np.linspace(-5000,0,bins+1)

counts = np.zeros((bins,bins))
scores = np.zeros((bins,bins))
for k in range(len(tr_score_histo)):
    for i in range(bins):
        if (tr_lik_histo[k] > xbins[i] and tr_lik_histo[k] < xbins[i+1]):
            for j in range(bins):
                if (sh_lik_histo[k] > ybins[j] and sh_lik_histo[k] < ybins[j+1]):
                    counts[i][j] += 1
                    scores[i][j] += tr_score_histo[k]
                    
tr_lik_bins = []
sh_lik_bins = []
mean_score = []
for i, row in enumerate(counts):
    for j in range(len(row)):
        if (counts[i][j]!=0):
            tr_lik_bins.append((xbins[i]+xbins[i+1])/2)
            sh_lik_bins.append((ybins[j]+ybins[j+1])/2)
            mean_score.append(scores[i][j]/counts[i][j])

my_cmap = plt.cm.jet
my_cmap.set_under('g',1)
plt.hist2d(tr_lik_bins, sh_lik_bins, bins=(xbins, ybins), weights=mean_score, cmap='coolwarm', cmin=1e-23, vmin=0, vmax=1)
plt.xlim(0,4000)
plt.ylim(-25000,0)
plt.xlabel('track reco likelihood', fontsize=20)
plt.ylabel('shower reco likelihood', fontsize=20)
plt.title('MC '+name+' with track lik > 0 & shower lik < 0', fontsize=20)
plt.colorbar(label='mean shower score')
plt.show()
'''
