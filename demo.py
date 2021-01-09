"""
Example script demonstrating molecular transformations, using logP as target.
Includes data preprocessing, training, and dreaming.
"""

import sys
import numpy as np
import yaml
import torch
import argparse
import data_loader
import mol_utils
from random import shuffle


sys.path.append('datasets')
from utils import change_str, make_dir
from train import train
from dream import dream

import time

def main():
    print('Start reading data file...')
    settings=yaml.load(open("settings.yml","r"))
    file_name = settings['data_preprocess']['smiles_file']
    num_train = settings['data']['num_train']
    num_dream = settings['data']['num_dream']
    lr_trains=settings['lr_train']
    lr_trains=[float(num) for num in lr_trains]
    lr_dreams=settings['lr_dream']
    lr_dreams=[float(num) for num in lr_dreams]
    batch_size=settings['training']['batch_size']
    num_mol = num_train

    if num_dream > num_train:
        num_mol = num_dream

    num_epochs = settings['training']['num_epochs']
    model_parameters = settings['model']
    dreaming_parameters = settings['dreaming']
    dreaming_parameters_str = '{}_{}'.format(dreaming_parameters['batch_size'],
                                                   dreaming_parameters['num_epochs'])
    training_parameters = settings['training']
    training_parameters_str = '{}_{}'.format(training_parameters['num_epochs'],
                                                training_parameters['batch_size'])
    data_parameters = settings['data']
    data_parameters_str = '{}_{}'.format(data_parameters['num_train'],
                                            data_parameters['num_dream'])
    upperbounds_tr = settings['upperbounds_tr']
    upperbounds_dr = settings['upperbounds_dr']
    props=settings['property_values']

    directory = change_str('dream_results/{}_{}'.format(data_parameters_str,
                                                        training_parameters_str))
    make_dir(directory)

    smiles_list = data_loader.read_smiles(file_name)
    selfies_alphabet, largest_selfies_len, smiles_alphabet, largest_smiles_len\
        = data_loader.get_selfie_and_smiles_info(smiles_list, file_name)
    selfies_list, smiles_list = \
        data_loader.get_selfie_and_smiles_encodings(smiles_list,num_mol)

    print('Finished acquiring data.\n')
    print('Calculating logP of all molecules...')
    prop_vals=mol_utils.logP_from_molecule(smiles_list)

    prop_vals=np.array(prop_vals)
    print('Finished calculating logP of all molecules.\n')

    print('Representation: SELFIES')
    alphabet=selfies_alphabet
    encoding_list=selfies_list
    largest_molecule_len=largest_selfies_len
    print('--> Creating one-hot encoding...')
    data = mol_utils.multiple_selfies_to_hot(encoding_list,
                                   largest_molecule_len,
                                   alphabet)
    print('    Finished creating one-hot encoding.\n')

    len_max_molec = data.shape[1]
    len_alphabet = data.shape[2]
    len_max_molec1Hot = len_max_molec * len_alphabet
    print(' ')
    print('Alphabet has ', len_alphabet, ' letters, largest molecule is ',
          len_max_molec, ' letters.')

    # if the system supports CUDA, utilize it for faster computation.
    parser = argparse.ArgumentParser(description='Set device')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    args = parser.parse_args()
    args.device = None

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu:0')

    # split data into training and test data.
    train_valid_test_size=[0.85, 0.15, 0.0]
    x = [i for i in range(len(data))]  # random shuffle input
    shuffle(x)
    data = data[x]
    prop_vals=prop_vals[x]

    data_dream = data[:num_dream]
    prop_dream = prop_vals[:num_dream]

    data = data[:num_train]
    prop_vals = prop_vals[:num_train]

    idx_traintest=int(len(data)*train_valid_test_size[0])
    idx_trainvalid=idx_traintest+int(len(data)*train_valid_test_size[1])
    data_train=data[0:idx_traintest]
    prop_vals_train=prop_vals[0:idx_traintest]

    data_test=data[idx_traintest:idx_trainvalid]
    prop_vals_test=prop_vals[idx_traintest:idx_trainvalid]

    # loop through each combination of hyperparameters:
    #   - upperbound for training and dreaming
    #   - learning rate for training and dreaming
    #   - target logP

    # loop through each training upperbound.
    for _,upperbound_tr in enumerate(upperbounds_tr):
        parent_dir1 = directory
        directory += change_str('/{}'.format(upperbound_tr))

        # loop through each training learning rate.
        for _,lr_train in enumerate(lr_trains):

            parent_dir2 = directory
            directory += change_str('/{}'.format(lr_train))
            make_dir(directory)

            t=time.clock()
            model = train(directory, args, model_parameters, len_max_molec1Hot,
                          upperbound_tr, data_train, prop_vals_train, data_test,
                          prop_vals_test, lr_train, num_epochs, batch_size)
            train_time = time.clock()-t

            # loop through each dreaming upperbound.
            for _,upperbound_dr in enumerate(upperbounds_dr):

                parent_dir3 = directory
                directory += change_str('/{}_{}'.format(upperbound_dr,
                                                        dreaming_parameters_str))
                make_dir(directory)
                print('upperbound: '+str(upperbound_dr))

                # loop through each dreaming learning rate.
                for _, lr_dream in enumerate(lr_dreams):

                    parent_dir4 = directory
                    directory += change_str('/{}'.format(lr_dream))
                    make_dir(directory)
                    print('lr_dream: '+str(lr_dream))

                    # loop through each target logP.
                    for _, prop in enumerate(props):

                        parent_dir5 = directory
                        directory += change_str('/{}'.format(prop))
                        make_dir(directory)

                        dream(directory, args, largest_molecule_len, alphabet,
                              model, train_time, upperbound_dr, data_dream,
                              prop_dream, prop, lr_train, lr_dream, num_train,
                              num_dream, dreaming_parameters)

                        directory = parent_dir5
                    directory = parent_dir4
                directory = parent_dir3
            directory = parent_dir2
        directory = parent_dir1

if __name__ == '__main__':
    main()
