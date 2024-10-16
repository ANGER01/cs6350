import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

from Helpers.helpers import get_data
import numpy as np
import pandas as pd


Bank_X_train, Bank_y_train, Bank_X_test, Bank_y_test = get_data("bank")



