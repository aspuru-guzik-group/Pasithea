"""
Utilities for various small tasks.
"""

import os
import matplotlib.pyplot as plt

def change_str(name):
    """Remove spaces, commas, semicolons, periods, brackets from given string
    and replace them with an underscore."""
    
    changed = ''
    for i in range(len(name)):
        if name[i]=='{' or name[i]=='}' or name[i]=='.' or name[i]==':' or name[i]==',' or name[i]==' ':
            changed += '_'
        elif name[i]=='\'':
            changed += ''
        else:
            changed += name[i]
    return changed

def make_dir(name):
    """Create a new directory."""
    
    if not os.path.exists(name):
        os.makedirs(name)


def closefig():
    """Clears and closes current instance of a plot."""
    plt.clf()
    plt.close()
    
