"""
Utilities for visualizing training and dreaming results.
"""
import matplotlib.pyplot as plt
from utils import closefig

def running_avg_test_loss(avg_test_loss, directory):
    """Plot running average test loss"""
    
    plt.figure()
    plt.plot(avg_test_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Running average test loss')
    name = name = directory +'/runningavg_testloss'
    plt.savefig(name)
    closefig()
    
def test_model_after_train(calc_train, real_vals_prop_train, 
               calc_test, real_vals_prop_test,
               directory, prop_name='logP'):
    """Scatter plot comparing ground truth data with the modelled data";
    includes both test and training data."""
    
    plt.figure()
    plt.scatter(calc_train,real_vals_prop_train,color='red',s=40, facecolors='none')
    plt.scatter(calc_test,real_vals_prop_test,color='blue',s=40, facecolors='none')
    plt.xlim(min(real_vals_prop_train)-0.5,max(real_vals_prop_train)+0.5)
    plt.ylim(min(real_vals_prop_train)-0.5,max(real_vals_prop_train)+0.5)
    plt.xlabel('Modelled '+prop_name)
    plt.ylabel('Computed '+prop_name)
    plt.title('Train set (red), test set (blue)')
    name = directory +'/test_model_after_training'
    plt.savefig(name)
    closefig()
    
def test_model_before_dream(trained_data_prop, computed_data_prop, 
                            directory, prop_name='logP'):
    """Scatter plot comparing ground truth data with modelled data"""
    
    plt.figure()
    plt.scatter(trained_data_prop, computed_data_prop)
    plt.xlabel('Modelled '+prop_name)
    plt.ylabel('Computed '+prop_name)
    name = directory +'/test_model_before_dreaming'
    plt.savefig(name)
    plt.show()
    closefig()
    
def prediction_loss(train_loss, test_loss, directory):
    """Plot prediction loss during training of model"""
    
    plt.figure()
    plt.plot(train_loss, color = 'red')
    plt.plot(test_loss, color = 'blue')
    plt.title('Prediction loss: training (red), test (blue)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    name = directory +'/predictionloss_test&train'
    plt.savefig(name)
    closefig()
    
def dreamed_histogram(prop_lst, prop, directory, prop_name='logP'):
    """Plot distribution of property values from a given list of values 
    (after transformation)"""
    
    plt.figure()
    plt.hist(prop_lst, density=True, bins=30)
    plt.ylabel(prop_name+' - around '+str(prop))
    name = directory + '/dreamed_histogram'
    plt.savefig(name)
    closefig()
    
def initial_histogram(prop_dream, directory, 
                      dataset_name='QM9', prop_name='logP'):
    """Plot distribution of property values from a given list of values
    (before transformation)"""
    
    plt.figure()
    plt.hist(prop_dream, density=True, bins=30)
    plt.ylabel(prop_name+' - '+dataset_name)
    name = directory+'/QM9_histogram'
    plt.savefig(name)
    closefig()