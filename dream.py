"""
Dreaming script.
"""

import torch
import time
import plot_utils
import mol_utils

from torch import nn
from mol_utils import edit_hot, lst_of_logP, multiple_hot_to_indices


def dream_model(model, prop, largest_molecule_len, alphabet, upperbound,
                data_train, lr, batch_size, num_epochs, display=True):
    """
    Trains in the inverse of the model with a single molecular input.
    Returns initial, final, and intermediate molecules/property values
    in the transformation;
    the percent of valid transformations;
    the list of loss terms during dreaming;
    and the list of epochs at which the molecule transformed during dreaming.
    """

    loss_prediction=[]

    # reshape for efficient parallelization
    data_train = data_train.reshape(data_train.shape[0],
                                    data_train.shape[1] * data_train.shape[2])

    # add random noise to one-hot encoding
    data_train_edit = edit_hot(data_train, upper_bound=upperbound)
    data_train_var=torch.autograd.Variable(data_train_edit, requires_grad=True)
    data_train_prop=torch.tensor([prop], dtype=torch.float)

    #initiailize list of intermediate property values and molecules
    interm_prop = []
    interm_mols = []

    epoch_transformed = []
    steps = 0
    valid_steps = 0

    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam([data_train_var], lr=lr)

    for epoch in range(num_epochs):

        # feedforward step
        calc_properties = model(data_train_var)

        # mean squared error between target and calculated property
        calc_properties = calc_properties.reshape(batch_size)
        criterion = nn.MSELoss()
        real_loss=criterion(calc_properties, data_train_prop)
        loss = torch.clamp(real_loss, min = 0., max = 50000.).double()

        # backpropagation step
        optimizer_encoder.zero_grad()
        loss.backward()
        optimizer_encoder.step()

        real_loss=loss.detach().numpy()
        loss_prediction.append(real_loss)


        if epoch%100==0:
            if display:
                print('epoch: ',epoch,', loss: ', real_loss)

        # convert one-hot encoding to SMILES molecule
        molecule_reshaped=torch.reshape(data_train_var,
                                        (1, largest_molecule_len,
                                         len(alphabet)))
        gathered_indices = multiple_hot_to_indices(molecule_reshaped)
        prop_of_mol, smiles_of_mol=lst_of_logP(gathered_indices, alphabet)

        if len(interm_prop)==0 or interm_prop[len(interm_prop)-1] != prop_of_mol[0]:

            # collect intermediate molecules
            interm_mols.append(smiles_of_mol[0])
            interm_prop.append(prop_of_mol[0])
            steps+=1
            epoch_transformed.append(epoch)

            if len(interm_prop)>1:

                # determine validity of transformation
                previous_prop = interm_prop[len(interm_prop)-2]
                current_prop = prop_of_mol[0]
                valid = (prop > previous_prop and current_prop > previous_prop) \
                        or (prop < previous_prop and current_prop < previous_prop)
                if valid:
                    valid_steps += 1

        if real_loss<1e-3:
            if display:
                print('Small loss, stop dreaming at epoch ', epoch)
            break

        if len(loss_prediction)>1000:
            if 0.99*loss_prediction[-900]<loss_prediction[-1]:
                if display:
                    print('Too small decrease, stop dreaming at epoch ', epoch)
                break

    # convert one-hot encoding to SMILES molecule
    molecule_reshaped=torch.reshape(data_train_var,
                                    (1, largest_molecule_len, len(alphabet)))
    gathered_indices = multiple_hot_to_indices(molecule_reshaped)
    prop_of_mol, smiles_of_mol=lst_of_logP(gathered_indices, alphabet)
    interm_mols.append(smiles_of_mol[0])
    interm_prop.append(prop_of_mol[0])

    percent_valid_transform = None
    if steps > 0:
        percent_valid_transform = valid_steps / steps *100

    return interm_prop, interm_mols, percent_valid_transform, loss_prediction, epoch_transformed



def dream(directory, args, largest_molecule_len, alphabet, model, train_time,
          upperbound, data_dream, prop_dream, prop,
          lr_train, lr_dream, num_train, num_dream, dreaming_parameters):
    """Dreaming procedure for a dataset of molecules. Saves the following
    results to file:
        - Summary of dreaming
        - All molecular transformations, mapping from initial to final
            molecule and property
        - Intermediate molecules for each transformation"""

    data_dream=torch.tensor(data_dream, dtype=torch.float, device=args.device)
    prop_dream = torch.tensor(prop_dream, dtype=torch.float, device=args.device)

    # plot initial distribution of property value in the dataset
    plot_utils.initial_histogram(prop_dream.numpy(), directory)
    avg1 = torch.mean(prop_dream).numpy()

    num_valid = 0
    num_unchanged = 0
    valid = False
    prop_lst = []
    interm = []
    transforms = []
    t= time.clock()
    for i in range(num_dream):
        print('Molecule #'+str(i))

        # convert one-hot encoding to SMILES molecule
        mol = data_dream[i].clone()
        gathered_mols=[]
        _,max_index=mol.max(1)
        gathered_mols.append(max_index.data.cpu().numpy().tolist())
        prop_of_mol,smiles_of_mol=mol_utils.lst_of_logP(gathered_mols, alphabet)

        mol1 = smiles_of_mol[0]
        mol1_prop = prop_of_mol[0]
        train_mol = torch.reshape(mol, (1, mol.shape[0], mol.shape[1]))

        # feed molecule into the inverse-model
        (track_prop, track_mol,
         percent_valid_interm,
         track_loss,
         epoch_transformed) = dream_model(model = model,
                                          prop=prop,
                                          largest_molecule_len=largest_molecule_len,
                                          alphabet=alphabet,
                                          upperbound = upperbound,
                                          data_train=train_mol,
                                          lr=lr_dream,
                                          **dreaming_parameters)

        # track and record results from dreaming
        prop_val = track_prop[len(track_prop)-1]
        mol2 = track_mol[len(track_mol)-1]
        valid = (prop > mol1_prop and prop_val > mol1_prop) or \
                (prop < mol1_prop and prop_val < mol1_prop)
        if valid:
            num_valid += 1
        if mol1_prop == prop_val or mol1==mol2:
            num_unchanged += 1
        percent_valid = num_valid*100/(i+1)
        percent_unchanged = num_unchanged*100/(i+1)
        percent_invalid = 100 - percent_valid -percent_unchanged
        transform = mol1+' --> '+mol2+', '+str(mol1_prop)+' --> '+str(prop_val)
        print('Transformation: ' +transform)
        print('Percent transformed in the direction of logP= '+ \
              str(prop)+': '+str(percent_valid)+'%')
        print('Percent transformed incorrectly: ' +str(percent_invalid)+'%')
        print('Percent not transformed: '+str(percent_unchanged)+'%')

        prop_lst.append(prop_val)
        transforms.append(transform)
        interm_tuple = ([mol1_prop]+track_prop, [mol1]+track_mol)
        interm.append(interm_tuple)

    dream_time = time.clock()-t

    # plot final distribution of property value after transformation
    plot_utils.dreamed_histogram(prop_lst, prop, directory)

    avg2 = sum(prop_lst)/len(prop_lst)

    # save a summary of the dreaming results to file
    name = directory + '/summary'
    f = open(name, "w+")

    f.write('Summary of dreaming:\n\n')
    f.write('Input upperbound='+str(upperbound) +'\n')
    f.write('Target logP='+str(prop)+'\n')
    f.write('Prediction lr='+str(lr_train)+'\n')
    f.write('Dreaming lr='+str(lr_dream)+'\n')
    f.write('Number of molecules trained:'+str(num_train)+'\n')
    f.write('Number of molecules dreamed:'+str(num_dream)+'\n')
    f.write('avg before dreaming: '+str(avg1)+'\n')
    f.write('avg after dreaming: '+str(avg2)+'\n')
    f.write('Percent unchanged: '+str(percent_unchanged)+'%\n')
    f.write('Percent adjusted toward target: '+str(percent_valid)+'%\n')
    f.write('Percent adjusted away from target: '+str(percent_invalid)+'%\n')
    f.write('Dreaming time: '+str(dream_time)+'\n')
    f.write('Training time: '+str(train_time)+'\n')

    f.close()

    # save list of all transformations to file
    name = directory +'/original_to_dream_mol'
    g = open(name, "w+")
    for t in transforms:
        g.write(t+'\n')
    g.close()

    # save intermediate molecules for each transformation to file
    name = directory + '/sampled_intermediate_mol'
    h1 = open(name, "w+")
    for i in range(len(interm)):
        h1.write('Sample '+str(i+1)+'\n')
        h1.write(str(interm[i][0])+'\n')
        h1.write(str(interm[i][1])+'\n')
    h1.close()


def mol_transform(mols, model, prop, largest_molecule_len, alphabet,
                  upperbound_dr, lr_dream, dreaming_parameters, plot=False):
    """Dreaming procedure for a set of molecules. Plots and saves to file
    the logP and loss evolution over number of epochs if desired."""

    for i, mol in enumerate(mols):
        mol = torch.reshape(mol, (1, mol.shape[0], mol.shape[1]))
        (track_prop, track_mol,
         percent_valid_interm,
         track_loss,
         epoch_transformed) = dream_model(model = model,
                                          prop=prop,
                                          largest_molecule_len=largest_molecule_len,
                                          alphabet=alphabet,
                                          upperbound = upperbound_dr,
                                          data_train=mol,
                                          lr=lr_dream,
                                          **dreaming_parameters,
                                          display=False)


        mol1_prop = track_prop[0]
        mol2_prop = track_prop[len(track_prop)-1]
        mol1 = track_mol[0]
        mol2 = track_mol[len(track_mol)-1]
        transform = mol1+' --> '+mol2+', '+str(mol1_prop)+' --> '+str(mol2_prop)
        print('Transformation '+ str(i+1)+': '+transform)
        print(track_mol)

        if plot:
            plot_utils.plot_transform(prop, track_mol, track_prop,
                                      epoch_transformed, track_loss)
