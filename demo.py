"""
Example script demonstrating molecular transformations, using logP as target.
The entire cycle - training and dreaming - is involved.
"""

import sys
sys.path.append('datasets')
import yaml
import data_loader
import time

from random import shuffle
from utils import change_str, make_dir, use_gpu
from train import train
from dream import dream


def main():
    # import hyperparameter and training settings from yaml
    settings = yaml.load(open("settings.yml", "r"))
    lr_trains = settings['lr_train']
    lr_trains = [float(num) for num in lr_trains]
    lr_dreams = settings['lr_dream']
    lr_dreams = [float(num) for num in lr_dreams]
    batch_size = settings['training']['batch_size']
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
    props = settings['property_values']

    num_train = settings['data']['num_train']
    num_dream = settings['data']['num_dream']

    num_mol = num_train

    if num_dream > num_train:
        num_mol = num_dream

    directory = change_str('dream_results/{}_{}'.format(data_parameters_str,
                                                        training_parameters_str))
    make_dir(directory)

    args = use_gpu()

    # data-preprocessing
    data, prop_vals, alphabet, len_max_molec1Hot, largest_molecule_len = \
        data_loader.preprocess(num_mol)

    # add stochasticity to data
    x = [i for i in range(len(data))]  # random shuffle input
    shuffle(x)
    data = data[x]
    prop_vals = prop_vals[x]

    data_dream = data[:num_dream]
    prop_dream = prop_vals[:num_dream]

    data_train, data_test, prop_vals_train, prop_vals_test \
        = data_loader.split_train_test(data, prop_vals, num_train, 0.85)

    # loop through each combination of hyperparameters:
    #   - upperbound for training and dreaming
    #   - learning rate for training and dreaming
    #   - target logP

    # loop through each training upperbound.
    for _, upperbound_tr in enumerate(upperbounds_tr):
        parent_dir1 = directory
        directory += change_str('/{}'.format(upperbound_tr))

        # loop through each training learning rate.
        for _, lr_train in enumerate(lr_trains):

            parent_dir2 = directory
            directory += change_str('/{}'.format(lr_train))
            make_dir(directory)

            t = time.clock()
            model = train(directory, args, model_parameters, len_max_molec1Hot,
                          upperbound_tr, data_train, prop_vals_train, data_test,
                          prop_vals_test, lr_train, num_epochs, batch_size)
            train_time = time.clock() - t

            # loop through each dreaming upperbound.
            for _, upperbound_dr in enumerate(upperbounds_dr):

                parent_dir3 = directory
                directory += change_str('/{}_{}'.format(upperbound_dr,
                                                        dreaming_parameters_str))
                make_dir(directory)
                print('upperbound: ' + str(upperbound_dr))

                # loop through each dreaming learning rate.
                for _, lr_dream in enumerate(lr_dreams):

                    parent_dir4 = directory
                    directory += change_str('/{}'.format(lr_dream))
                    make_dir(directory)
                    print('lr_dream: ' + str(lr_dream))

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
