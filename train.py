"""
Training script.
"""

import numpy as np
import torch
import os
import plot_utils

from torch import nn
from random import shuffle
from mol_utils import edit_hot
from utils import change_str
from model import fc_model



def train_model(parent_dir, directory, args, model,
                upperbound, data_train, data_train_prop, data_test,
                data_test_prop, lr_enc, num_epochs, batch_size):
    """Train the model"""

    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam(model.parameters(), lr=lr_enc)

    # reshape for efficient parallelization
    data_train=torch.tensor(data_train, dtype=torch.float, device=args.device)
    data_test=torch.tensor(data_test, dtype=torch.float, device=args.device)
    reshaped_data_train = torch.reshape(data_train,
                                        (data_train.shape[0],
                                         data_train.shape[1]*data_train.shape[2]))
    reshaped_data_test = torch.reshape(data_test,
                                       (data_test.shape[0],
                                        data_test.shape[1]*data_test.shape[2]))

    # add random noise to one-hot encoding
    reshaped_data_test_edit = edit_hot(reshaped_data_test, upperbound)

    data_train_prop=torch.tensor(data_train_prop,
                                 dtype=torch.float, device=args.device)
    data_test_prop=torch.tensor(data_test_prop,
                                dtype=torch.float, device=args.device)

    test_loss=[]
    train_loss=[]
    avg_test_loss=[]
    min_loss = 1

    for epoch in range(num_epochs):

        # add stochasticity to the training
        x = [i for i in range(len(reshaped_data_train))]  # random shuffle input
        shuffle(x)
        reshaped_data_train  = reshaped_data_train[x]
        data_train_prop = data_train_prop[x]
        reshaped_data_train_edit = edit_hot(reshaped_data_train,
                                            upper_bound=upperbound)

        for batch_iteration in range(int(len(reshaped_data_train_edit)/batch_size)):

            current_smiles_start, current_smiles_stop = \
                batch_iteration * batch_size, (batch_iteration + 1) * batch_size

            # slice data into batches
            curr_mol=reshaped_data_train_edit[current_smiles_start : \
                                              current_smiles_stop]
            curr_prop=data_train_prop[current_smiles_start : \
                                      current_smiles_stop]

            # feedforward step
            calc_properties = model(curr_mol)
            calc_properties=torch.reshape(calc_properties,[len(calc_properties)])

            # mean-squared error between calculated property and modelled property
            criterion = nn.MSELoss()
            real_loss=criterion(calc_properties, curr_prop)

            loss = torch.clamp(real_loss, min = 0., max = 50000.).double()

            # backpropagation step
            optimizer_encoder.zero_grad()
            loss.backward()
            optimizer_encoder.step()

        # calculate train set
        calc_train_set_property = model(reshaped_data_train_edit)
        calc_train_set_property=torch.reshape(calc_train_set_property,
                                              [len(calc_train_set_property)])
        criterion = nn.MSELoss()
        real_loss_train=criterion(calc_train_set_property, data_train_prop)
        real_loss_train_num=real_loss_train.detach().cpu().numpy()

        # calculate test set
        calc_test_set_property = model(reshaped_data_test_edit)
        criterion = nn.MSELoss()
        calc_test_set_property=torch.reshape(calc_test_set_property,
                                             [len(calc_test_set_property)])
        real_loss_test=criterion(calc_test_set_property, data_test_prop)
        real_loss_test_num=real_loss_test.detach().cpu().numpy()


        print('epoch: '+str(epoch)+' - avg loss: '+\
              str(np.mean(real_loss_train_num))+', testset: '+\
              str(np.mean(real_loss_test_num)))

        test_loss.append(real_loss_test_num)
        train_loss.append(real_loss_train_num)

        if real_loss_test_num < min_loss:
            min_loss = real_loss_test_num
            torch.save(model.state_dict(), parent_dir)

            print('Test loss decrease, model saved to file')

        # stopping criterion: compare the running test loss averages over 90 epochs
        if len(test_loss)>=100:
            avg = sum(test_loss[len(test_loss)-90:len(test_loss)])
            avg_test_loss.append(avg)

            print(avg_test_loss)

            if len(avg_test_loss)>=50 and avg>avg_test_loss[len(avg_test_loss)-40]:
                print('Train loss is increasing, stop training')

                # plot training results
                real_vals_prop_train=data_train_prop.detach().cpu().numpy()
                real_vals_prop_test=data_test_prop.detach().cpu().numpy()

                calc_train=calc_train_set_property.detach().cpu().numpy()
                calc_test=calc_test_set_property.detach().cpu().numpy()

                plot_utils.running_avg_test_loss(avg_test_loss, directory)
                plot_utils.test_model_after_train(calc_train, real_vals_prop_train,
                                                  calc_test,real_vals_prop_test,
                                                  directory)
                plot_utils.prediction_loss(train_loss, test_loss, directory)
                break

def load_model(file_name, args, len_max_molec1Hot, model_parameters):
    """Load existing model state dict from file"""

    model = fc_model(len_max_molec1Hot, **model_parameters).to(device=args.device)
    model.load_state_dict(torch.load(file_name))
    model.eval()
    return model

def train(directory, args, model_parameters, len_max_molec1Hot, upperbound,
          data_train, prop_vals_train, data_test, prop_vals_test, lr_train,
          num_epochs, batch_size):
    name = change_str(directory)+'/model.pt'

    if os.path.exists(name):
        model = load_model(name, len_max_molec1Hot, model_parameters)
        print('Testing model...')
        test_model(directory, args, model,
                   data_train, prop_vals_train, upperbound)
    else:
        print('No models saved in file with current settings.')
        model = fc_model(len_max_molec1Hot, **model_parameters).to(device=args.device)
        model.train()

        print('len(data_train): ',len(data_train))
        print("start training")

        train_model(name, directory, args, model, upperbound,
                    data_train, prop_vals_train, data_test, prop_vals_test,
                    lr_train, num_epochs, batch_size)

        model = fc_model(len_max_molec1Hot, **model_parameters).to(device=args.device)
        model.load_state_dict(torch.load(name))
        model.eval()
        print('Testing model...')
        test_model(directory, args, model,
                   data_train, prop_vals_train, upperbound)
        print('finished training and testing, now start dreaming :)\n\n\n')

    return model

def test_model(directory, args, model, data, data_prop, upperbound):
    """Test model to ensure it is sufficiently trained before dreaming."""

    test_data = torch.tensor(data, dtype=torch.float, device=args.device)
    computed_data_prop = torch.tensor(data_prop, device=args.device)

    # reshape for efficient parallelization
    test_data = test_data.reshape(test_data.shape[0],
                                  test_data.shape[1] * test_data.shape[2])

    # add random noise to one-hot encoding with specified upperbound
    test_data_edit = edit_hot(test_data, upperbound)

    # feedforward step
    trained_data_prop = model(test_data_edit)
    trained_data_prop = trained_data_prop.reshape(data.shape[0]).clone().detach().numpy()

    # compare ground truth data to modelled data
    plot_utils.test_model_before_dream(trained_data_prop, computed_data_prop,
                                       directory)
