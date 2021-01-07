"""
Example script for visualizing the distribution shifts from dreaming results.
(Particularly, the results from molecular transformations.)
"""
import matplotlib.pyplot as plt
from utils import change_str
    
def read_logP(file_name):
    """Returns the list of all initial property values and the list of all
    final property values, given a file of strings that map from initial
    to final property values through molecular dreaming."""
    
    init_prop = [] # initialize list of initial property values
    final_prop = [] # initialize list of final property values
    
    f= open(file_name, "r")
    for line in f:
        prop1 = ''
        prop2 = ''
        pos = len(line)-3
        while pos >= 0 and line[pos] != '>':
            prop1 = line[pos] + prop1
            pos -= 1
            
        pos -= 4
        while pos >= 0 and line[pos] != ' ':
            prop2 = line[pos] + prop2
            pos -= 1
        
        final_prop.append(float(prop1))
        init_prop.append(float(prop2))
        
    f.close()
    return init_prop, final_prop

def read_mol(file_name):
    """Returns the list of all initial molecules and the list of all final
    molecules, given a file of strings that map from initial to final
    molecules through molecular dreaming."""
    
    init_mol = [] #initialize list of initial molecules
    final_mol = [] #initialize list of final molecules
    
    f = open(file_name, "r")
    for line in f:
        mol1 = ''
        mol2 = ''
        pos = 0
        while pos <= len(line)-1 and line[pos] != ' ':
            mol1 += line[pos]
            pos += 1
        pos += 4
        while pos <= len(line)-1 and line[pos] != ',':
            mol2 += line[pos]
            pos += 1
        init_mol.append(mol1)
        final_mol.append(mol2)
    f.close()
    return init_mol, final_mol

def plot_distribution(upperbound_tr, upperbound_dr, prediction_lr, 
                      inverted_lr, init_prop, 
                      final_prop_pos, final_prop_neg, target_pos, target_neg, 
                      init_mol, final_mol_pos, final_mol_neg):
    
    """Plots three histograms showing the initial distribution and
    the distribution after positive and negative optimization, given
    the set of the hyperparameters.
    """
    
    n_bins = 20
    range_lst = [-4, 4]
    
    plt.hist(init_prop, n_bins, density=True, alpha=0.8, label='QM9', 
             color = 'red', range = range_lst, histtype='step', fill=False)
    plt.hist(final_prop_pos, n_bins, density=True, alpha=0.5, 
             label='Target logP = '+str(target_pos), 
             color = "#003366", range = range_lst)
    plt.hist(final_prop_neg, n_bins, density=True, alpha=0.5, 
             label='Target logP = '+str(target_neg), 
             color = '#006600', range = range_lst)
    
    # print numerical stats information for the distribution shift
    print('{}_{}_{}_{}_distribution'.format(change_str(str(upperbound_tr)), 
                                            change_str(str(upperbound_dr)),
                                            change_str(str(prediction_lr)), 
                                            change_str(str(inverted_lr))))
    print('Max logP in qm9:',max(init_prop), 
          "Mol:",init_mol[init_prop.index(max(init_prop))])
    print('Max logP after pos dream:',max(final_prop_pos),
          "Mol:",final_mol_pos[final_prop_pos.index(max(final_prop_pos))])
    print('Max logP after neg dream:',max(final_prop_neg),
          "Mol:",final_mol_neg[final_prop_neg.index(max(final_prop_neg))])
    print('Min logP in qm9:',min(init_prop),
          "Mol:",init_mol[init_prop.index(min(init_prop))])
    print('Min logP after pos dream:',min(final_prop_pos),
          "Mol:",final_mol_pos[final_prop_pos.index(min(final_prop_pos))])
    print('Min logP after neg dream:',min(final_prop_neg),
          "Mol:",final_mol_neg[final_prop_neg.index(min(final_prop_neg))])
    print('Mean logP in qm9:',sum(init_prop)/len(init_prop))
    print('Mean logP in pos dream:',sum(final_prop_pos)/len(final_prop_pos))
    print('Mean logP in neg dream:',sum(final_prop_neg)/len(final_prop_neg))
    
    print()
    
    
    #plt.hist([qm9_logP, dream_logP_pos, dream_logP_neg], n_bins, 
    #         density=True, alpha=0.5, label=colors)
    plt.legend(prop={'size': 10})
    plt.title('\ntraining upperbound = '+str(upperbound_tr)+\
              ', dreaming upperbound = ' +str(upperbound_dr)+\
                  ', \nprediction lr = '+\
              str(prediction_lr)+', inverted lr = '+str(inverted_lr))
    plt.xlabel('LogP')
    plt.ylabel('Density')
    plt.ylim(0, 0.8)
    plt.savefig(folder+\
                '/{}_{}_{}_{}_distributions.pdf'.format(change_str(str(upperbound_tr)),
                                                        change_str(str(upperbound_dr)),
                                                        change_str(str(prediction_lr)), 
                                                        change_str(str(inverted_lr))))
    plt.show()
    
def read_and_plot_distribution(prop_pos, prop_neg, upperbound_tr,
                               upperbound_dr, prediction_lr, inverted_lr):
    """Reads from file that corresponds to the given hyperparameters.
    Plots three histograms showing the initial distribution and
    the distribution after positive and negative optimization, given
    the set of the hyperparameters."""
    
    directory = \
    "{}/10000_500_10000_1/{}/{}/{}_1_10000/{}/".format(folder,
                                                      change_str(str(upperbound_tr)),
                                                      change_str(str(prediction_lr)), 
                                                      change_str(str(upperbound_dr)), 
                                                      change_str(str(inverted_lr)))
    
    file_name = directory + str(prop_pos)+"/original_to_dream_mol"
    init_prop_pos, final_prop_pos = read_logP(file_name)
    init_mol_pos, final_mol_pos = read_mol(file_name)
    file_name = directory + str(prop_neg)+"/original_to_dream_mol"
    init_prop_neg, final_prop_neg = read_logP(file_name)
    init_mol_neg, final_mol_neg = read_mol(file_name)
    if init_prop_pos[0] == init_prop_neg[0]:
        plot_distribution(upperbound_tr, upperbound_dr, prediction_lr, inverted_lr, 
                          init_prop_pos, final_prop_pos, final_prop_neg, 
                          prop_pos, prop_neg, 
                          init_mol_pos, final_mol_pos, final_mol_neg)
    
if __name__ == '__main__':
    folder = "dream_results"
    read_and_plot_distribution(6, -6, 0.95, 0.90, 1e-06, 0.01)
    read_and_plot_distribution(6, -6, 0.95, 0.75, 1e-06, 0.01)
    read_and_plot_distribution(6, -6, 0.95, 0.5, 1e-06, 0.01)
    read_and_plot_distribution(6, -6, 0.95, 0.25, 1e-06, 0.01)
    