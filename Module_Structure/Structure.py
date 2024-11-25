import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas as pd

# Ouvrir le fichier donnée aéro 

Actual_Path = os.getcwd()
Actual_Path = Actual_Path.split("MDOHypersonic")[0]
filePath = str(Actual_Path + "MDOHypersonic\\Module_Aero\\hyper.dbg")

with open(filePath,'r') as file:
    Data = file.read()

# Localiser Intrados Demi-voilure
network_2_start = None
network_2_end = None
lines = Data.split("\n")
for i, line in enumerate(lines):
    if line== " NETWORK           2":
        network_2_start = i
    if line== " NETWORK           3":
        network_2_end = i-2   
        break

# Importer données voilure
network_2_start = None
network_2_end = None
lines = Data.split("\n")
for i, line in enumerate(lines):
    if line== " NETWORK           2":
        network_2_start = i
    if line== " NETWORK           3":
        network_2_end = i-2   
        break