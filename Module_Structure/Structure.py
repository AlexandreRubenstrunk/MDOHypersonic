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
Pos_Foyer_Aile = 13.41 #Déclaré au quart de corde

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
df_Panel_Data[['#', 'net', 'row', 'col', 'center x', 'center y', 'center z', 'normal x', 'normal y', 'normal z', 'area']] = df_Panel_Data[['#', 'net', 'row', 'col', 'center x', 'center y', 'center z', 'normal x', 'normal y', 'normal z', 'area']].apply(pd.to_numeric, errors='coerce')
# df_Panel_Data_Utile = df_Panel_Data[['#', 'center x', 'center y', 'center z', 'area', 'normal x', 'normal y', 'normal z']]

# Traitement des données Aérodynamiques
# Extraction données aéro 

CP_Data_Raw = []
CP_Data_Raw_start = None
CP_Data_Raw_end = None
Inclinaison_Bonne = False
lines = Data.split("\n")
for i, line in enumerate(lines):
    if "ALPHA" in line and " 1.0000" in line and "BETA" in line:
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
df_CP_Data[['#', 'net', 'C', 'E', 'center x', 'center y', 'center z', 'CP', 'cosdel', 'delta']] = df_CP_Data[['#', 'net', 'C', 'E', 'center x', 'center y', 'center z', 'CP', 'cosdel', 'delta']].apply(pd.to_numeric, errors='coerce')
df_CP_Data_Utile = df_CP_Data[['#', 'CP']]

# Fusion des données

Merged_Data = pd.merge(df_Panel_Data, df_CP_Data_Utile, on='#')

# Passage du nuage de point de Cp à F

Forces_Data = []
for _, row in Merged_Data.iterrows():
    ref = int(row['#'])
    net = int(row['net'])
    R = int(row['row'])
    col = int(row['col'])
    x, y, z = float(row['center x']), float(row['center y']), float(row['center z'])
    area = float(row['area'])
    Cp = float(row['CP'])
    
    P = 0.5 * Rho * Valtitude**2 * Cp - PsInf
    
    Fx = P * area * float(row['normal x'])
    Fy = P * area * float(row['normal y'])
    Fz = P * area * float(row['normal z'])
    
    Forces_Data.append([ref, net, R, col, x, y, z, Cp, Fx, Fy, Fz])

columns = ['#', 'net', 'row', 'col', 'x', 'y', 'z', 'Cp', 'Fx', 'Fy', 'Fz']
df_Forces_Data = pd.DataFrame(Forces_Data, columns=columns)
df_Forces_Data[['#', 'net', 'row', 'col', 'x', 'y', 'z', 'Cp', 'Fx', 'Fy', 'Fz']] = df_Forces_Data[['#', 'net', 'row', 'col', 'x', 'y', 'z', 'Cp', 'Fx', 'Fy', 'Fz']].apply(pd.to_numeric, errors='coerce')

# Déterminer le nombre et la position des sections

network_to_filter = 2  # Correspondant à l'extrados, même discrétisation à l'intrados
filtered_data = df_Forces_Data[df_Forces_Data['net'] == network_to_filter]
Nbre_sections = filtered_data['y'].unique()
Valeurs_sections = len(Nbre_sections)

# Calculer dans chaques sections l'effort tranchant et la répartition sur les longerons

Pos_LBA_rel_corde = 0.15 #valeur arbitraires de calcul, proche avion civile
Pos_LBF_rel_corde = 0.75 #valeur arbitraires de calcul, proche avion civile
Force_Voilure = []
for Pos_y_Section in Nbre_sections :
    network_to_filter = 2 
    Data_Section_Extrados = df_Forces_Data[(df_Forces_Data['net'] == network_to_filter) & (df_Forces_Data['y'] == Pos_y_Section)]
    network_to_filter = 3 
    Data_Section_Intrados = df_Forces_Data[(df_Forces_Data['net'] == network_to_filter) & (df_Forces_Data['y'] == Pos_y_Section)]

    Effort_Tranchant_Section = Data_Section_Extrados['Fz'].sum() + Data_Section_Intrados['Fz'].sum()
    Moment_x_Section = (Data_Section_Extrados['x'] * Data_Section_Extrados['Fz']).sum() + (Data_Section_Intrados['x'] * Data_Section_Intrados['Fz']).sum()
    Pos_LBA = Data_Section_Extrados['x'].min() + Pos_LBA_rel_corde*(Data_Section_Extrados['x'].max()-Data_Section_Extrados['x'].min())
    Pos_LBF = Data_Section_Extrados['x'].min() + Pos_LBF_rel_corde*(Data_Section_Extrados['x'].max()-Data_Section_Extrados['x'].min())

    Force_longeron_arr = ( Moment_x_Section-Pos_LBA * Effort_Tranchant_Section ) / ( Pos_LBF - Pos_LBA )
    Force_longeron_av = Effort_Tranchant_Section - Force_longeron_arr

    Force_Voilure.append([Pos_y_Section, Effort_Tranchant_Section, Pos_LBA, Force_longeron_av, Pos_LBF, Force_longeron_arr, 0])

columns = ['Pos y Section', 'Eff tranchant', 'Pos LBA', 'Fz LBA', 'Pos LBF', 'Fz LBF', 'Mf section']
df_Force_Voilure = pd.DataFrame(Force_Voilure, columns=columns)
df_Force_Voilure[['Pos y Section', 'Eff tranchant', 'Pos LBA', 'Fz LBA', 'Pos LBF', 'Fz LBF', 'Mf section']] = df_Force_Voilure[['Pos y Section', 'Eff tranchant', 'Pos LBA', 'Fz LBA', 'Pos LBF', 'Fz LBF', 'Mf section']].apply(pd.to_numeric, errors='coerce')

for i in range(len(df_Force_Voilure) - 1, -1, -1):
    if i == len(df_Force_Voilure) - 1:
        df_Force_Voilure.loc[i, 'Mf section'] = 0
    else:
        Moment_temp = 0
        for j in range(len(df_Force_Voilure) - 1, i, -1):
            delta_y = df_Force_Voilure.loc[j , 'Pos y Section'] - df_Force_Voilure.loc[i, 'Pos y Section']
            Moment_temp = Moment_temp + df_Force_Voilure.loc[j , 'Eff tranchant'] * delta_y
        df_Force_Voilure.loc[i, 'Mf section'] = Moment_temp

portance = 2 * df_Force_Voilure['Eff tranchant'].sum()
print(portance)

# Visualisation
# Tracer l'effort tranchant en fonction de l'envergure
plt.figure(figsize=(12, 6))

# Graphique 1 : Effort tranchant vs envergure
plt.subplot(1, 2, 1)
plt.plot(df_Force_Voilure['Pos y Section'], df_Force_Voilure['Eff tranchant'], marker='o', label="Effort Tranchant")
plt.title("Effort Tranchant en fonction de l'envergure")
plt.xlabel("Position en y (envergure) [m]")
plt.ylabel("Effort Tranchant [N]")
plt.grid(True)
plt.legend()

# Graphique 2 : Moment de flexion vs envergure
plt.subplot(1, 2, 2)
plt.plot(df_Force_Voilure['Pos y Section'], df_Force_Voilure['Mf section'], marker='o', color='red', label="Moment de Flexion")
plt.title("Moment de Flexion en fonction de l'envergure")
plt.xlabel("Position en y (envergure) [m]")
plt.ylabel("Moment de Flexion [Nm]")
plt.grid(True)
plt.legend()

# Afficher les graphiques
plt.tight_layout()
plt.show()

print("fin")