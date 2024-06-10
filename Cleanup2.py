import os
import pandas as pd
import numpy as np

#read in the dataset
year = '2021'
brfss_2021_dataset = pd.read_csv(r"C:\Users\Roccas\Documents\TMU Data\CIND820 Big Data Analytics Project\LLCP2021XPT\diabetes_health_indicators_BRFSS2021.csv")

brfss_2021_dataset.drop('Hello', inplace=True, axis=1)

brfss_2021_dataset.to_csv(r"C:\Users\Roccas\Documents\TMU Data\CIND820 Big Data Analytics Project\LLCP2021XPT\diabetes_health_indicators_BRFSS2021.csv")