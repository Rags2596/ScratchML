import numpy as np
import pandas as pd
import os
import sys
from sklearn import datasets
from sklearn.model_selection import train_test_split



from knn import KNN

data = datasets.load_iris()

print(data)