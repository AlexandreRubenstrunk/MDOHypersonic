import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import os
import pandas as pd

# Constantes pour les calculs atmosphériques
T0 = 288.15  # Température au niveau de la mer (K)
R = 287  # Constante des gaz parfaits pour l'air (J/(kg*K))
Gamma = 1.4  # Rapport des chaleurs spécifiques (Cp/Cv)
MmAir = 0.0289644  # Masse molaire de l'air (kg/mol)
Rgp = 8.3144621  # Constante universelle des gaz parfaits (J/(mol*K))
Donnee_Altitude = 18288  # Altitude de vol à étudier (m) correspondant au FL600

# Récupération du chemin du fichier de données
Actual_Path = os.getcwd()
Actual_Path = Actual_Path.split("MDOHypersonic")[0]
filePath = str(Actual_Path + "MDOHypersonic\\Module_Aero\\hyper.out")

# Lecture du fichier contenant les données aérodynamiques
with open(filePath, 'r') as file:
    Data = file.read()

# Initialisation des variables pour extraire les données du fichier
Flight_Data_Raw = []
Flight_Data_Raw_start = None
Flight_Data_Raw_end = None
Data_found = False

# Parcours du fichier ligne par ligne pour localiser les sections utiles
line = Data.split("\n")
for i, lin in enumerate(line):
    if "Mach=" in lin and not Data_found:
        lin_Mach = i # Ligne contenant les informations du Mach et de la Sref
        Data_found = True
    if lin == " SOLUTIONS FOR COMPLETE VEHICLE IN BODY AXES":
        Flight_Data_Raw_start = i + 2  # Ligne de début des données utiles
    if lin == " SOLUTIONS FOR COMPLETE VEHICLE IN WIND AXES":
        Flight_Data_Raw_end = i - 1  # Ligne de fin des données utiles
        break

# Extraction des paramètres Mach et surface référentielle (Sref)
lineMach = line[lin_Mach].strip().split()
Donnee_Mach = float(lineMach[1])
Donnee_Sref = float(lineMach[3])

# Fonction pour calculer les conditions ISA (International Standard Atmosphere)
def ISA(Altitude, Mach):
    """
    Calcule les conditions atmosphériques à une altitude donnée selon le modèle ISA.

    Paramètres:
        Altitude (float): Altitude en mètres.
        Mach (float): Nombre de Mach du vol.

    Retourne:
        tuple: Pression statique (Pa), densité (kg/m^3), température (K), vitesse (m/s).
    """
    H_Base = 11000  # Altitude limite entre la troposphère et la stratosphère (m)
    if Altitude < H_Base:
        Pbase = 101325  # Pression au niveau de la mer (Pa)
        Tbase = 288.15  # Température au niveau de la mer (K)
        TAltitude = T0 - 0.0065 * Altitude
        PsInf = Pbase * (TAltitude / Tbase) ** (-9.81 * MmAir / (Rgp * (-0.0065)))
    else:
        Pbase = 22632.1  # Pression de base au-dessus de 11 km (Pa)
        TAltitude = 216.65  # Température constante au-dessus de 11 km (K)
        PsInf = Pbase * math.exp(-9.81 * MmAir * (Altitude - H_Base) / (Rgp * TAltitude))

    Rho = PsInf / (R * TAltitude)  # Densité (kg/m^3)
    Valtitude = Mach * (Gamma * R * TAltitude) ** 0.5  # Vitesse du son (m/s)

    return PsInf, Rho, TAltitude, Valtitude

# Calcul des paramètres atmosphériques pour l'altitude et le Mach donnés
PsInf, Rho, TAltitude, Valtitude = ISA(Donnee_Altitude, Donnee_Mach)

# Extraction et mise en forme des données aérodynamiques
for i in range(Flight_Data_Raw_start, Flight_Data_Raw_end):
    Flight_Data_Raw.append(line[i])

Flight_Data = []
for lin in Flight_Data_Raw:
    row = lin.strip().split()
    Fz_normalise = 0.5 * Rho * Valtitude**2 * float(row[5])  # Normalisation des forces avec le Cl de l'appareil complet
    row.append(Fz_normalise)
    Flight_Data.append(row)

# Création d'un DataFrame pour organiser les données aérodynamiques
columns = ['#', 'alpha', 'beta', 'cx', 'cy', 'cz', 'cmx', 'cmy', 'cmz', 'cln']
df_Flight_Data = pd.DataFrame(Flight_Data, columns=columns)
df_Flight_Data[columns] = df_Flight_Data[columns].apply(pd.to_numeric, errors='coerce')

# Obtention du chemin du fichier de données géométriques
Actual_Path = os.getcwd()
Actual_Path = Actual_Path.split("MDOHypersonic")[0]
filePath = str(Actual_Path + "MDOHypersonic\\Module_Aero\\hyper.dbg")

# Lecture du fichier contenant les données géométriques
with open(filePath, 'r') as file:
    Data2 = file.read()

# Initialisation des variables pour extraire les données géométriques
Panel_Data_Raw = []
Panel_Data_Raw_start = None
Panel_Data_Raw_end = None
lines = Data2.split("\n")

# Recherche des sections utiles dans le fichier
for i, line in enumerate(lines):
    if line == " PANEL DATA":
        Panel_Data_Raw_start = i + 2  # Ligne de début des données utiles
    if line == " NETWORK AREAS AND CENTERS":
        Panel_Data_Raw_end = i - 1  # Ligne de fin des données utiles
        break

# Extraction des données géométriques brutes
for i in range(Panel_Data_Raw_start, Panel_Data_Raw_end):
    Panel_Data_Raw.append(lines[i])

# Mise en forme des données géométriques dans une liste structurée
Panel_Data = []
for line in Panel_Data_Raw:
    row = line.strip().split()
    Panel_Data.append(row)

# Création d'un DataFrame pour organiser les données géométriques
columns = ['#', 'net', 'row', 'col', 'center x', 'center y', 'center z', 'normal x', 'normal y', 'normal z', 'area']
df_Panel_Data = pd.DataFrame(Panel_Data, columns=columns)
df_Panel_Data[columns] = df_Panel_Data[columns].apply(pd.to_numeric, errors='coerce')

# Calcul de la force normalisée (Fz) sur chaque panneau

# Extraction du coefficient de portance normalisé pour un angle d'attaque donné (ici 6°)
Cl_normalise_calcul = Flight_Data[6][9]  # Coefficient de portance normalisé pour α = 6°

# Initialisation de la liste pour stocker les données calculées
Forces_Data = []

# Parcours de chaque ligne du DataFrame contenant les données des panneaux
for _, row in df_Panel_Data.iterrows():
    """
    Pour chaque panneau :
    - Récupère ses propriétés (identifiant, position, surface).
    - Calcule la force normale Fz en fonction du coefficient normalisé et de la surface.
    - Stocke les résultats dans une liste pour construire un nouveau DataFrame.
    """

    # Identification du panneau
    ref = int(row['#'])         # Référence unique du panneau
    net = int(row['net'])       # Indique si le panneau est sur l'extrados ou l'intrados
    R = int(row['row'])         # Ligne (row) du panneau
    col = int(row['col'])       # Colonne (col) du panneau

    # Position du centre du panneau dans l'espace (x, y, z)
    x, y, z = float(row['center x']), float(row['center y']), float(row['center z'])

    # Surface du panneau
    area = float(row['area'])

    # Calcul de la force normale Fz sur le panneau
    Fz = Cl_normalise_calcul * area  # Fz = coefficient * surface

    # Ajout des données calculées dans la liste
    Forces_Data.append([ref, net, R, col, x, y, z, Fz])

# Création d'un DataFrame pour organiser les forces calculées
columns = ['#', 'net', 'row', 'col', 'x', 'y', 'z', 'Fz']
df_Forces_Data = pd.DataFrame(Forces_Data, columns=columns)
df_Forces_Data[columns] = df_Forces_Data[columns].apply(pd.to_numeric, errors='coerce')

# Filtrer les données pour un réseau spécifique (extrados)
network_to_filter = 2  # Réseau pour l'extrados
filtered_data = df_Forces_Data[df_Forces_Data['net'] == network_to_filter]

# Identifier les sections en fonction de leur position en y
Nbre_sections = filtered_data['y'].unique()
Valeurs_sections = len(Nbre_sections)

# Paramètres de positionnement des longerons (valeurs proches d'un avion civil)
Pos_LBA_rel_corde = 0.20
Pos_LBF_rel_corde = 0.75

# Calculer l'effort tranchant et la répartition des forces par section
Force_Voilure = []

for Pos_y_Section in Nbre_sections:
    """
    Itère sur chaque section de la voilure définie par sa position en y (envergure) 
    et calcule les forces appliquées ainsi que les caractéristiques des longerons (LBA et LBF).
    
    Paramètres :
    ------------
    - Pos_y_Section : Position en y correspondant à une section donnée.

    Résultats ajoutés à la liste `Force_Voilure` :
    -----------------------------------------------
    - Position de la section (Pos_y_Section)
    - Effort tranchant sur la section (Effort_Tranchant_Section)
    - Positions et hauteurs des longerons (Pos_BA, Pos_LBA, Pos_BF, Pos_LBF, Hauteur_LBA, Hauteur_LBF)
    - Forces appliquées sur les longerons avant et arrière (Force_longeron_av, Force_longeron_arr)
    """

    # Filtrer les données pour les extrados et intrados de la section actuelle
    Data_Section_Extrados = df_Forces_Data[
        (df_Forces_Data['net'] == 2) & (df_Forces_Data['y'] == Pos_y_Section)
    ]
    Data_Section_Intrados = df_Forces_Data[
        (df_Forces_Data['net'] == 3) & (df_Forces_Data['y'] == Pos_y_Section)
    ]

    # Calculer l'effort tranchant total sur la section
    Effort_Tranchant_Section = 0.5 * (
        Data_Section_Extrados['Fz'].sum() + Data_Section_Intrados['Fz'].sum()
    )
    
    # Calculer le moment de flexion autour de l'axe x
    Moment_x_Section = (
        (Data_Section_Extrados['x'] * Data_Section_Extrados['Fz']).sum() +
        (Data_Section_Intrados['x'] * Data_Section_Intrados['Fz']).sum()
    )

    # Déterminer les positions relatives des longerons avant et arrière
    Pos_LBA = Data_Section_Extrados['x'].min() + Pos_LBA_rel_corde * (
        Data_Section_Extrados['x'].max() - Data_Section_Extrados['x'].min()
    )
    Pos_LBF = Data_Section_Extrados['x'].min() + Pos_LBF_rel_corde * (
        Data_Section_Extrados['x'].max() - Data_Section_Extrados['x'].min()
    )

    # Initialiser des variables pour indiquer si les positions des longerons sont trouvées
    Valeure_LBA = False
    Valeure_LBF = False

    # Parcourir les points de la section pour calculer les hauteurs des longerons
    for i in range(len(Data_Section_Extrados)):
        # Calcul de la hauteur au niveau du longeron avant (LBA)
        if not Valeure_LBA and Data_Section_Extrados['x'].iloc[i] >= Pos_LBA:
            Valeure_LBA = True
            Hauteur_LBA = (
                (Data_Section_Extrados['z'].iloc[i - 1] - Data_Section_Intrados['z'].iloc[i - 1]) +
                2 * (Data_Section_Extrados['z'].iloc[i] - Data_Section_Intrados['z'].iloc[i - 1]) *
                ((Pos_LBA - Data_Section_Intrados['x'].iloc[i - 1]) /
                 (Data_Section_Intrados['x'].iloc[i] - Data_Section_Intrados['x'].iloc[i - 1]))
            )

        # Calcul de la hauteur au niveau du longeron arrière (LBF)
        if not Valeure_LBF and Data_Section_Extrados['x'].iloc[i] >= Pos_LBF:
            Valeure_LBF = True
            Hauteur_LBF = (
                (Data_Section_Extrados['z'].iloc[i - 1] - Data_Section_Intrados['z'].iloc[i - 1]) +
                2 * (Data_Section_Extrados['z'].iloc[i] - Data_Section_Intrados['z'].iloc[i - 1]) *
                ((Pos_LBF - Data_Section_Intrados['x'].iloc[i - 1]) /
                 (Data_Section_Intrados['x'].iloc[i] - Data_Section_Intrados['x'].iloc[i - 1]))
            )

    # Calculer les forces appliquées sur les longerons
    Force_longeron_arr = (
        Moment_x_Section - Pos_LBA * Effort_Tranchant_Section
    ) / (Pos_LBF - Pos_LBA)
    Force_longeron_av = Effort_Tranchant_Section - Force_longeron_arr

    # Déterminer les positions avant (Pos_BA) et arrière (Pos_BF) de la section
    Pos_BA = Data_Section_Extrados['x'].min()
    Pos_BF = Data_Section_Extrados['x'].max()

    # Ajouter les résultats dans la liste finale
    Force_Voilure.append([
        Pos_y_Section, Effort_Tranchant_Section, Pos_BA, Pos_LBA, Hauteur_LBA, Force_longeron_av,
        Pos_BF, Pos_LBF, Hauteur_LBF, Force_longeron_arr, 0, 0
    ])

# Création du DataFrame pour les forces
columns = ['Pos y Section', 'Eff tranchant', 'Pos BA', 'Pos LBA', 'Haut LBA', 'Fz LBA', 'Pos BF', 'Pos LBF', 'Haut LBF', 'Fz LBF', 'Mf section', 'Portance']
df_Force_Voilure = pd.DataFrame(Force_Voilure, columns=columns)
df_Force_Voilure = df_Force_Voilure.apply(pd.to_numeric, errors='coerce')

# Calcul des moments de flexion et de la portance pour chaque section de la voilure
for i in range(len(df_Force_Voilure) - 1, -1, -1):
    """
    Ce bloc calcule :
    - Le moment de flexion (`Mf section`) pour chaque section en remontant depuis l'extrémité (bout de la voilure).
    - La portance cumulée à partir du bout de l'aile.

    Processus :
    ------------
    - Si la section est la dernière (bout de la voilure), son moment de flexion est nul,
      et la portance à son bout est égale à la portance locale.
    - Pour les autres sections, le moment de flexion est calculé en intégrant
      l'effort tranchant sur la distance entre les sections.
    - La portance cumulative est mise à jour en remontant vers la racine de l'aile.
    """

    if i == len(df_Force_Voilure) - 1:
        # Cas particulier : dernière section (bout de la voilure)
        df_Force_Voilure.loc[i, 'Mf section'] = 0  # Moment de flexion nul au bout
        df_Force_Voilure.loc[i, 'Portance bout'] = df_Force_Voilure.loc[i, 'Portance']  # Portance locale = portance au bout
    else:
        # Initialisation du moment temporaire pour cette section
        Moment_temp = 0

        # Calcul du moment de flexion dû aux efforts tranchants des sections en aval
        for j in range(len(df_Force_Voilure) - 1, i, -1):
            delta_y = df_Force_Voilure.loc[j, 'Pos y Section'] - df_Force_Voilure.loc[i, 'Pos y Section']
            Moment_temp += df_Force_Voilure.loc[j, 'Eff tranchant'] * delta_y

        # Mise à jour du moment de flexion pour la section actuelle
        df_Force_Voilure.loc[i, 'Mf section'] = Moment_temp

        # Mise à jour de la portance cumulative au bout
        df_Force_Voilure.loc[i, 'Portance bout'] = (
            df_Force_Voilure.loc[i + 1, 'Portance bout'] + df_Force_Voilure.loc[i, 'Eff tranchant']
        )

# Calcul de la portance totale
portance = 2 * df_Force_Voilure['Eff tranchant'].sum()
print(portance)

# Visualisation des résultats
plt.figure(figsize=(12, 6))

# Graphique 1 : Effort tranchant vs envergure
plt.subplot(1, 2, 1)
plt.plot(df_Force_Voilure['Pos y Section'], df_Force_Voilure['Portance bout'], marker='o', label="Effort Tranchant")
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

# Affichage des graphiques
plt.tight_layout()
plt.show()

# Définition de la contrainte maximale du matériau utilisé pour les longerons
Sigma_max = 900e6  # Contrainte maximale pour le tungsten haute densité WHD11 en Pascal

# Liste pour stocker les données calculées pour la structure de la voilure
Structure_Voilure = []
for i in range(len(df_Force_Voilure)):
    """
    Pour chaque section définie dans `df_Force_Voilure`, calcule les dimensions 
    et les caractéristiques des longerons (LBA et LBF) en fonction de la contrainte maximale 
    admissible et des moments de flexion appliqués.
    
    Étapes principales :
    ---------------------
    - Calcul de la largeur des longerons nécessaire pour résister aux efforts appliqués.
    - Déduction des sections transversales des longerons (LBA et LBF).
    - Extraction des positions en y (envergure) et x (corde) des longerons pour chaque section.
    """

    # Calcul de la largeur des longerons en fonction du moment de flexion et des hauteurs des longerons
    Largeur_Longeron = (
        3 * df_Force_Voilure['Mf section'].iloc[i] * 
        ((1 / Hauteur_LBA * Hauteur_LBA) + (1 / Hauteur_LBF * Hauteur_LBF)) / 
        Sigma_max
    )
    
    # Calcul des sections transversales des longerons avant (LBA) et arrière (LBF)
    Section_LBA = Hauteur_LBA * Largeur_Longeron
    Section_LBF = Hauteur_LBF * Largeur_Longeron
    
    # Extraction des coordonnées de la section courante
    y = df_Force_Voilure['Pos y Section'].iloc[i]  # Position en y (envergure)
    x_LBA = df_Force_Voilure['Pos LBA'].iloc[i]    # Position en x du longeron avant
    x_LBF = df_Force_Voilure['Pos LBF'].iloc[i]    # Position en x du longeron arrière

    # Ajouter les résultats calculés à la liste
    Structure_Voilure.append([y, x_LBA, Section_LBA, x_LBF, Section_LBF])

# Définir les colonnes du DataFrame correspondant
columns = ['Pos y Section', 'Pos LBA', 'Section LBA', 'Pos LBF', 'Section LBF']
df_Structure_Voilure = pd.DataFrame(Structure_Voilure, columns=columns)
df_Structure_Voilure = df_Structure_Voilure.apply(pd.to_numeric, errors='coerce')

# Initialiser la figure
plt.figure(figsize=(10, 8))

# Tracer les contours des points (x, y) de df_Forces_Voilure correspondant a la première et la dernière discrétisation sur la section
plt.plot(
    df_Force_Voilure['Pos BA'], df_Force_Voilure['Pos y Section'], 
    df_Force_Voilure['Pos BF'], df_Force_Voilure['Pos y Section'],
    linestyle='-', color='blue', label="Contour du profil discrétisé", alpha=0.7
)

# Ajouter les longerons LBA et LBF
# Normaliser les valeurs des sections pour appliquer la colormap
norm = Normalize(vmin=min(df_Structure_Voilure[['Section LBA', 'Section LBF']].min()),
                 vmax=max(df_Structure_Voilure[['Section LBA', 'Section LBF']].max()))
cmap = cm.coolwarm

# Plot des points correspondant au longeron de bord d'attaque (LBA)
scatter_LBA = plt.scatter(
    df_Structure_Voilure['Pos LBA'], df_Structure_Voilure['Pos y Section'], 
    c=df_Structure_Voilure['Section LBA'], cmap=cmap, norm=norm, label="Longeron BA", s=50
)

# Plot des points correspondant au longeron de bord de fuite (LBF)
scatter_LBF = plt.scatter(
    df_Structure_Voilure['Pos LBF'], df_Structure_Voilure['Pos y Section'], 
    c=df_Structure_Voilure['Section LBF'], cmap=cmap, norm=norm, label="Longeron BF", s=50
)

# Ajouter une barre de couleur
cbar = plt.colorbar(scatter_LBA, orientation="vertical")
cbar.set_label("Section du longeron [m²]", fontsize=12)

# Configurer l'affichage
plt.title("Représentation du contour discret et des longerons de voilure", fontsize=14)
plt.xlabel("Position x [m]", fontsize=12)
plt.ylabel("Position y (envergure) [m]", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Afficher le graphique
plt.show()

print("fin")
