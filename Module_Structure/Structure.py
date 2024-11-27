import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
# import sys
# print("\n".join(sys.path))

# from ..Module_Propu.PropuHyperso3Shocks import ISA

# P,Rho,T=ISA(0)

#Données du fichier donnée aéro à transferer entre les dossiers
T0 = 288.15         #Sea Level temp in Kelvin
R = 287
Gamma = 1.4
MmAir = 0.0289644      #Molar masses of the air
Rgp = 8.3144621         #Universal cst of perfect gazs

Donnee_Altitude = 18288 #Donnée à importer de aéro
Donnee_Mach = 5 #Donnée à importer de aéro

def ISA(Altitude,Mach):
    H_Base = 11000
    if Altitude<H_Base:
        Pbase = 101325
        Tbase = 288.15
        TAltitude = T0-0.0065*Altitude
        PsInf = Pbase*(TAltitude/Tbase)**(-9.81*MmAir/(Rgp*(-0.0065)))
    else:
        Pbase = 22632.1
        TAltitude = 216.65
        PsInf = Pbase*math.exp(-9.81*MmAir*(Altitude-H_Base)/(Rgp*TAltitude))

    Rho = PsInf/(R*TAltitude)
    Valtitude = Mach*(Gamma*R*TAltitude)**0.5

    return PsInf,Rho, TAltitude, Valtitude 
 
PsInf, Rho, TAltitude, Valtitude = ISA(Donnee_Altitude, Donnee_Mach)

Actual_Path = os.getcwd()
Actual_Path = Actual_Path.split("MDOHypersonic")[0]
filePath = str(Actual_Path + "MDOHypersonic\\Module_Aero\\hyper.dbg")

with open(filePath,'r') as file:
    Data = file.read()

# Traitement des données Géométriques
# Importer les données Géo

Panel_Data_Raw = []
Panel_Data_Raw_start = None
Panel_Data_Raw_end = None
lines = Data.split("\n")
for i, line in enumerate(lines):
    if line== " PANEL DATA":
        Panel_Data_Raw_start = i+2
    if line== " NETWORK AREAS AND CENTERS":
        Panel_Data_Raw_end = i-1   
        break

# Mise en forme des données Géo

for i in range(Panel_Data_Raw_start, Panel_Data_Raw_end):
   Panel_Data_Raw.append(lines[i])

Panel_Data = []
for line in Panel_Data_Raw:
    row = line.strip().split()
    Panel_Data.append(row)

columns = ['#', 'net', 'row', 'col', 'center x', 'center y', 'center z', 'normal x', 'normal y', 'normal z', 'area']
df_Panel_Data = pd.DataFrame(Panel_Data, columns=columns)

# Traitement des données Aérodynamiques
# Extraction données aéro 

CP_Data_Raw = []
CP_Data_Raw_start = None
CP_Data_Raw_end = None
Inclinaison_Bonne = False
lines = Data.split("\n")
for i, line in enumerate(lines):
    if "ALPHA" in line and " 2.0000" in line and "BETA" in line:
        CP_Data_Raw_start = i+2
        Inclinaison_Bonne = True
    if " NETWORK FORCES AND MOMENTS" in line and Inclinaison_Bonne == True :
         CP_Data_Raw_end = i   
         break

# Mise en forme des données Aéro

for i in range(CP_Data_Raw_start, CP_Data_Raw_end):
   CP_Data_Raw.append(lines[i])

CP_Data = []
for line in CP_Data_Raw:
    row = line.strip().split()
    CP_Data.append(row)

columns = ['#', 'net', 'C', 'E', 'center x', 'center y', 'center z', 'CP', 'cosdel', 'delta']
df_CP_Data = pd.DataFrame(CP_Data, columns=columns)

# Passage de Cp à P, besoin Pinf, Rhoinf et Vinf

# def pression_aerodynamique(Cp, rho, v, pinf):
#     return 0.5 * rho * v**2 * Cp - pinf


# P = pression_aerodynamique(Cp, rho, v)
print("ok")