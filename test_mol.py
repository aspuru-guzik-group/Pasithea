"""
Test molecules on Pasithea. Visualize molecular transformations.
"""

import data_loader
import yaml
from mol_utils import multiple_selfies_to_hot
from utils import make_dir, change_str, use_gpu
from train import train, load_model
from dream import mol_transform
import torch


def main():
    # import hyperparameter and training settings from yaml
    print('Start reading data file...')
    settings=yaml.load(open("test_mol_settings.yml","r"))
    test = settings['test_model']
    plot = settings['plot_transform']
    mols = settings['mols']
    file_name = settings['data_preprocess']['smiles_file']
    lr_train=settings['lr_train']
    lr_train=float(lr_train)
    lr_dream=settings['lr_dream']
    lr_dream=float(lr_dream)
    batch_size=settings['training']['batch_size']
    num_epochs = settings['training']['num_epochs']
    model_parameters = settings['model']
    dreaming_parameters = settings['dreaming']

    training_parameters = settings['training']
    training_parameters_str = '{}_{}'.format(training_parameters['num_epochs'],
                                             training_parameters['batch_size'])
    data_parameters = settings['data']
    data_parameters_str = '{}_{}'.format(data_parameters['num_train'],
                                         data_parameters['num_dream'])

    upperbound_tr = settings['upperbound_tr']
    upperbound_dr = settings['upperbound_dr']
    prop=settings['property_value']

    num_train = settings['data']['num_train']
    num_dream = settings['data']['num_dream']

    num_mol = num_train

    if num_dream > num_train:
        num_mol = num_dream

    directory = change_str('dream_results/{}_{}/{}/{}' \
                           .format(data_parameters_str,
                                   training_parameters_str,
                                   upperbound_tr,
                                   lr_train))
    make_dir(directory)

    args = use_gpu()

    # data-preprocessing
    data, prop_vals, alphabet, len_max_molec1Hot, largest_molecule_len = \
        data_loader.preprocess(num_mol, file_name)

    if test:
        data_train, data_test, prop_vals_train, prop_vals_test \
            = data_loader.split_train_test(data, prop_vals, num_train, 0.85)

        # also need to test if the model is fine
        model = train(directory, args, model_parameters, len_max_molec1Hot,
                      upperbound_tr, data_train, prop_vals_train, data_test,
                      prop_vals_test, lr_train, num_epochs, batch_size)
    else:
        model = load_model(directory+'/model.pt', args, len_max_molec1Hot,
                           model_parameters)

    # convert from SMILES to SELFIES
    selfies_lst, _ = data_loader.get_selfie_and_smiles_encodings(mols)

    # convert from SELFIES to one-hot encoding
    mols = multiple_selfies_to_hot(selfies_lst,
                                   largest_molecule_len,
                                   alphabet)
    mols = torch.tensor(mols, dtype=torch.float, device=args.device)

    # molecular transformations
    mol_transform(mols, model, prop, largest_molecule_len, alphabet,
                  upperbound_dr, lr_dream, dreaming_parameters, plot)


if __name__ == '__main__':
    main()
