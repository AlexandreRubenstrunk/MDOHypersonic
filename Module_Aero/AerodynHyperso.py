import numpy as np
import math
import matplotlib.pyplot as plt
import os
import subprocess
import re
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from matplotlib.tri import Triangulation

class Body:
    def __init__(self):
        self.Lenght = float()
        self.Lenght_Cabine = float()
        self.Lenght_Nose = float()
        self.Diameter = float()
        self.OffSet_Nose = float()
        self.OffSet_Tail = float()
        self.PosX = float(0)
        self.PosY = float(0)
        self.PosZ = float(0)
        self.Number = float(1)

class Wing:
    def __init__(self):
        self.Sref = float()
        self.Xref = float()
        self.AR = float()
        self.TR = float()
        self.AF = str()
        self.PosX = float()
        self.PosZ = float()
        self.Sweep = float()

class Aircraft:

    def __init__(self):
        self.Name = str()
        self.Fuselage = Body()
        self.Wing = Wing()
        self.Nacelles = Body()

    def OpenAvion(Name):
        """This function is made to open a TXT file representing an Aircraft.\n
        filePath: Acces path to the txt file\n
        It return an Aircraft with all the values added.
        """

        Actual_Path = os.getcwd()
        Actual_Path = Actual_Path.split("MDOHypersonic")[0]
        filePath = str(Actual_Path + "MDOHypersonic\\Aircraft\\" + str(Name) + ".txt")

        with open(filePath,'r') as file:
            Data = file.read()
        Parts1Line = Data.split("\n\n")
        Parts = []
        for Part1Line in Parts1Line:
            Part = Part1Line.split("\n")
            Parts.append(Part)
        
        Plane = Aircraft()

        parties = {
        "FUSELAGE": Plane.Fuselage,
        "WING": Plane.Wing,
        # "HT": Plane.HT,
        # "VT": Plane.VT,
        "NACELLES": Plane.Nacelles
        }

        for Part in Parts:
            if Part[0] in parties:
                for Param in Part[1:]:
                    Split = Param.split(":")
                    Attribut = Split[0].strip()
                    Value = Split[1].strip()
                    if hasattr(parties[Part[0]], Attribut):
                        if "AF" in Attribut:
                            setattr(parties[Part[0]], Attribut, Value)
                        else:
                            setattr(parties[Part[0]], Attribut, float(Value))

        return Plane

def PointBody(Body:Body,NB_CF, NB_Point_CF):
    """ This function define the carateristique point of the fuselage\n
    Avion: Is the studied aircraft\n
    NB_CF: Is the number of countour\n
    NB_Point_CF: Is the number of point per countour\n
    It return a numpy containing of the point sorted by Countour and Point number. Each point is define in 3 dimension."""

    Point=np.zeros((NB_CF,NB_Point_CF,3))

    if NB_CF >=4:
        Additionnal = (NB_CF-4)
        Disciminant = Additionnal % 3

        Number_Controur_Per_Part = Additionnal//3

        if Disciminant == 0:
            Front_Cabine_Indexe = Number_Controur_Per_Part + 1
            Rear_Cabin_Indexe = Front_Cabine_Indexe + Number_Controur_Per_Part + 1
            Rear_Fuselage_Indexe  = Rear_Cabin_Indexe + Number_Controur_Per_Part + 1
        if Disciminant == 1:
            Front_Cabine_Indexe = Number_Controur_Per_Part + 1
            Rear_Cabin_Indexe = Front_Cabine_Indexe + Number_Controur_Per_Part + 2
            Rear_Fuselage_Indexe  = Rear_Cabin_Indexe + Number_Controur_Per_Part + 1
        if Disciminant == 2:
            Front_Cabine_Indexe = Number_Controur_Per_Part + 2
            Rear_Cabin_Indexe = Front_Cabine_Indexe + Number_Controur_Per_Part + 2
            Rear_Fuselage_Indexe = Rear_Cabin_Indexe + Number_Controur_Per_Part + 1

        Point[0,:,0] = Body.PosX
        Point[0,:,2] = Body.OffSet_Nose + Body.PosZ

        Point[Rear_Fuselage_Indexe,:,2] = Body.OffSet_Tail + Body.PosZ
        Point[Rear_Fuselage_Indexe,:,0] = Body.Lenght + Body.PosX

        Point[Front_Cabine_Indexe,:,0] = Body.Lenght_Nose + Body.PosX
        Point[Rear_Cabin_Indexe,:,0] = Body.Lenght_Nose+Body.Lenght_Cabine + Body.PosX

        Teta = math.pi/(NB_Point_CF-1)

        for i in range(NB_Point_CF):
            Point[Front_Cabine_Indexe,i,1] = (Body.Diameter*math.sin(i*Teta)/2) + Body.PosY
            Point[Rear_Cabin_Indexe,i,1] = (Body.Diameter*math.sin(i*Teta)/2) + Body.PosY

            Point[Front_Cabine_Indexe,i,2] = (Body.Diameter*math.cos(i*Teta)/2) + Body.PosZ
            Point[Rear_Cabin_Indexe,i,2] = (Body.Diameter*math.cos(i*Teta)/2) + Body.PosZ

            for j in range(Front_Cabine_Indexe):
                if Front_Cabine_Indexe >1:
                    Indexe_Norm = (j/(Front_Cabine_Indexe-1))
                    Point[j,i,:] = (Point[0,i,:]*(1-Indexe_Norm)+Point[Front_Cabine_Indexe,i,:]*Indexe_Norm)

            for j in range(Front_Cabine_Indexe+1,Rear_Cabin_Indexe):
                if Rear_Cabin_Indexe-Front_Cabine_Indexe>1:
                    Indexe_Norm = ((j-Front_Cabine_Indexe)/((Rear_Cabin_Indexe-Front_Cabine_Indexe)-1))
                    Point[j,i,:] = (Point[Front_Cabine_Indexe,i,:]*(1-Indexe_Norm)+Point[Rear_Cabin_Indexe,i,:]*Indexe_Norm)

            for j in range(Rear_Cabin_Indexe+1,Rear_Fuselage_Indexe):
                if Rear_Fuselage_Indexe-Rear_Cabin_Indexe>1:
                    Indexe_Norm = ((j-Rear_Cabin_Indexe)/((Rear_Fuselage_Indexe-Rear_Cabin_Indexe)-1))
                    Point[j,i,:] = (Point[Rear_Cabin_Indexe,i,:]*(1-Indexe_Norm)+Point[Rear_Fuselage_Indexe,i,:]*Indexe_Norm)

    return Point

def Point2String(PT, STR, NB_C, NB_Point_C):
    max_longueur = 79

    for i in range(NB_C):
        STR += "\n"
        for j in range(NB_Point_C):
            point_str = f"{PT[i, j, 0]:.8E} {PT[i, j, 1]:.8E} {PT[i, j, 2]:.8E}"

            if len(STR) > 0 and len(STR) + len(point_str) + 1 > max_longueur:
                STR += "\n"

            STR += point_str + " "
    return STR

def CreationFuselage(Avion:Aircraft,NB_CF:int, NB_Point_CF:int):
    """ This function aime to create the string that define the fuselage\n
    Avion: Is the studied aircraft\n
    NB_CF: Is the number of countour\n
    NB_Point_CF: Is the number of point per countour\n"""

    STR = str("'Body'\n1\t" + str(NB_CF) + "\t"+ str(NB_Point_CF) + "\t0\t0 0 0\t0 0 0\t1 1 1\t1")

    PF = PointBody(Avion.Fuselage,NB_CF, NB_Point_CF)

    STR = Point2String(PF,STR,NB_CF,NB_Point_CF)

    return STR,PF

def OpenAF(Avion):
    Actual_Path = os.getcwd()
    Actual_Path = Actual_Path.split("MDOHypersonic")[0]
    AFPath = str(Actual_Path + "MDOHypersonic\\Aircraft\\Profile\\" + Avion.Wing.AF + ".txt")

    with open(AFPath,'r') as file:
        DataAF = file.read()
    DataAF = DataAF.split("\n")
    AFcoordinat = np.array([list(map(float, item.split('\t'))) for item in DataAF])
    return AFcoordinat

def CreationPointWing(Avion:Aircraft,NB_CW):
    AFCoordinat = OpenAF(Avion)
    PointWing_Upper = np.zeros((NB_CW,AFCoordinat.shape[0],3))
    PointWing_Lower = np.zeros((NB_CW,AFCoordinat.shape[0],3))

    Dim_0 = AFCoordinat[:,0]
    Dim_1 = AFCoordinat[:,1]

    InterAF = interp1d(Dim_0, Dim_1, kind='linear', fill_value="extrapolate")

    Span = (Avion.Wing.Sref*Avion.Wing.AR)**0.5

    for i in range(NB_CW):

        y_position = (Span / 2) * np.sin(i * np.pi / (2 * (NB_CW - 1)))
        
        PointWing_Upper[i, :, 1] = y_position
        PointWing_Lower[i, :, 1] = y_position

        Span = (Avion.Wing.Sref*Avion.Wing.AR)**0.5
        MAC = Avion.Wing.Sref/Span
        Croot = 2 * MAC/(1+Avion.Wing.TR)
        for j in range(AFCoordinat.shape[0]):
            PosSweep = PointWing_Lower[i,j,1]*math.tan(math.radians(Avion.Wing.Sweep))

            PointWing_Upper[i,j,0] =  Avion.Wing.PosX + j*Croot*(1-(1-Avion.Wing.TR)*2*PointWing_Upper[i,j,1]/Span)/(AFCoordinat.shape[0]-1)
            PointWing_Upper[i,j,0] = PointWing_Upper[i,j,0]+PosSweep 
            PointWing_Upper[i,j,2] =  Avion.Wing.PosZ + Croot*(1-(1-Avion.Wing.TR)*2*PointWing_Upper[i,j,1]/Span)*InterAF(j/(AFCoordinat.shape[0]-1))
            
            PointWing_Lower[i,j,0] = Avion.Wing.PosX + j*Croot*(1-(1-Avion.Wing.TR)*2*PointWing_Upper[i,j,1]/Span)/(AFCoordinat.shape[0]-1)
            PointWing_Lower[i,j,0] = PosSweep+PointWing_Lower[i,j,0]
            PointWing_Lower[i,j,2] = Avion.Wing.PosZ - Croot*(1-(1-Avion.Wing.TR)*2*PointWing_Upper[i,j,1]/Span)*InterAF(j/(AFCoordinat.shape[0]-1))
    
    return PointWing_Upper,PointWing_Lower,AFCoordinat.shape[0]

def CreationWing(Avion,NB_CW):
    PW_Upper,PW_Lower,NB_Point_CW = CreationPointWing(Avion,NB_CW)

    STR_Upper = str("\n'WING-UPPER'\n1\t" + str(NB_CW) + "\t"+ str(NB_Point_CW) + "\t0\t0 0 0\t0 0 0\t1 1 1\t1")
    STR_Upper = Point2String(PW_Upper,STR_Upper,NB_CW,NB_Point_CW)

    STR_Lower = str("\n'WING-LOWER'\n1\t" + str(NB_CW) + "\t"+ str(NB_Point_CW) + "\t0\t0 0 0\t0 0 0\t1 1 1\t1")
    STR_Lower = Point2String(PW_Lower,STR_Lower,NB_CW,NB_Point_CW)  

    STR_Wing = STR_Upper + STR_Lower
    PW = np.concatenate((PW_Upper,PW_Lower))

    return STR_Wing, PW

def CreationNacelles(Avion:Aircraft,NB_CN,NB_Point_CN,Number):

    STR = str("\n'BODY'\n" +str(Number) + "\t" + str(NB_CN) + "\t"+ str(NB_Point_CN) + "\t0\t0 0 0\t0 0 0\t1 1 1\t1")

    PN = PointBody(Avion.Nacelles,NB_CN, NB_Point_CN)

    STR = Point2String(PN,STR,NB_CN,NB_Point_CN)

    return STR,PN

def CFDFileWing(Avion:Aircraft,WingPoint):
    Actual_Path = os.getcwd()
    Actual_Path = Actual_Path.split("MDOHypersonic")[0]
    AFPath = str(Actual_Path + "MDOHypersonic\\Aircraft\\CFDFile\\" + Avion.Name + "CFDWing.txt")

    midpoint = WingPoint.shape[0]//2

    Upper = WingPoint[:midpoint,:,:]
    Lower = WingPoint[midpoint:,:,:]

    Root = np.concatenate(Upper[0,:,:],Lower[0,:,:][::-1])
    Tip = np.concatenate(Upper[-1,:,:],Lower[-1,:,:][::-1])

    CfdFilePoint = np.concatenate(Root,Tip)

    STR = str()
    for i in range(CfdFilePoint.shape[0]):
        str+=str("1\t" + str(i+1) + "\t" + CfdFilePoint[i])


    with open(AFPath,"w") as File:
        File.write(STR)
    

def FileCreation(Avion,Mach,Alphas,NB_CF=4, NB_Point_CF=5, NB_CW=6, CFD=False):
    """This function aime to create the file needed to compute an aircraft with Hyper.\n
    Avion: Is the studied aircraft\n
    Mach: Is the studied mach number\n
    Alphas: Is a Tab containening the values of AoA to be studied\n
    NB_CF: Is the number of countour\n
    NB_Point_CF: Is the number of point per countour\n
    """

    STRTotal = "Analyse module aero\n"
    FuselageSTR,FuselagePoint = CreationFuselage(Avion,NB_CF, NB_Point_CF)
    WingSTR,WingPoint = CreationWing(Avion,NB_CW)
    NacellesSTR,NacellesPoint = CreationNacelles(Avion,NB_CF,NB_Point_CF,2)

    if CFD == True:
        CFDFileWing(Avion,WingPoint)


    if __name__=="__main__":
        PlotAircraft(FuselagePoint,WingPoint,NacellesPoint)
        PlotAircraftMesh(FuselagePoint,WingPoint,NacellesPoint)

    STRTotal+=FuselageSTR
    STRTotal+=WingSTR
    STRTotal+=NacellesSTR

    Actual_Path = os.getcwd()
    Actual_Path = Actual_Path.split("MDOHypersonic")[0]
    FilePath = str(Actual_Path + "MDOHypersonic\\Module_Aero\\hyperFolder\\Analyse_Module_Aero.wgs")

    with open(FilePath,"w") as File:
        File.write(STRTotal)

    Span = (Avion.Wing.Sref*Avion.Wing.AR)**0.5
    MAC = Avion.Wing.Sref/Span

    STRinp = str("&hyp  title='Analyse module aero',\nwgsFileName= 'hyperFolder\\Analyse_Module_Aero.wgs' ,\ncmethods=7,1,1,7,\nemethods=5,5,5,5,\nmach=")
    STRinp += str(str(Mach) + ", sref=" + str(Avion.Wing.Sref) + ", cbar=" + str(MAC) + ",\n")
    STRinp += str("alpha="+",".join(map(str, Alphas)) + ",\n")
    STRinp += str("xref=" + str(Avion.Wing.Xref) + ", span=" + str(Span) +"/\n")

    INPFilePath = str(Actual_Path + "MDOHypersonic\\Module_Aero\\hyperFolder\\Analyse_Module_Aero.inp")

    with open(INPFilePath,"w") as File:
        File.write(STRinp)

    return

def PlotAircraftMesh(PointFuselage, Pointwing, PointNacelles):
    """Plots all points of the defined aircraft with symmetry along the Y-axis.\n
    PointFuselage: Numpy array containing the points sorted by contour and point number with coordinates in 3D
    Pointwing: Same format for the wing points
    PointNacelles: Same format for the nacelle points"""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def plot_symmetric_points(points, color, label):
        x = points[:, :, 0].flatten()
        y = points[:, :, 1].flatten()
        z = points[:, :, 2].flatten()

        # Plot original side
        ax.scatter(x, y, z, label=label, color=color)
        
        # Plot mirrored side by inverting y-coordinates
        ax.scatter(x, -y, z, color=color)

    # Plot each part of the aircraft with symmetry as points
    plot_symmetric_points(PointFuselage, 'b', 'Fuselage')
    plot_symmetric_points(Pointwing, 'r', 'Wing')
    plot_symmetric_points(PointNacelles, 'g', 'Nacelles')

    # Setting up labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.axis("equal")
    plt.show()

def PlotAircraft(PointFuselage, Pointwing, PointNacelles):
    """Plots the aircraft points as solid surfaces with symmetry along the Y-axis.\n
    PointFuselage, Pointwing, PointNacelles: Numpy arrays with fuselage, wing, and nacelle points in 3D coordinates."""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def plot_symmetric_cylinder(points, color):
        x = points[:, :, 0]
        y = points[:, :, 1]
        z = points[:, :, 2]

        # Original cylinder surface
        ax.plot_surface(x, y, z, color=color, alpha=0.7, shade=True)

        # Mirrored cylinder surface
        ax.plot_surface(x, -y, z, color=color, alpha=0.7, shade=True)

    def plot_symmetric_surface(points, color):
        x = points[:, :, 0].flatten()
        y = points[:, :, 1].flatten()
        z = points[:, :, 2].flatten()

        # Wing surface triangulation
        tri = Triangulation(x, y)
        ax.plot_trisurf(x, y, z, triangles=tri.triangles, color=color, alpha=0.7, shade=True)
        ax.plot_trisurf(x, -y, z, triangles=tri.triangles, color=color, alpha=0.7, shade=True)

    # Plot fuselage and nacelles as cylindrical surfaces
    plot_symmetric_cylinder(PointFuselage, 'b')
    plot_symmetric_cylinder(PointNacelles, 'g')

    # Plot wings as triangulated surfaces
    plot_symmetric_surface(Pointwing, 'r')

    # Manually create legend entries
    proxy_fuselage = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, label='Fuselage')
    proxy_wing = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Wing')
    proxy_nacelles = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Nacelles')

    # Adding the legend with proxy objects
    ax.legend(handles=[proxy_fuselage, proxy_wing, proxy_nacelles])

    # Setting up labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.axis("equal")
    plt.show(block=False)

def RunCalculation():

    Actual_Path = os.getcwd()
    Actual_Path = Actual_Path.split("MDOHypersonic")[0]
    Hyperpath = os.path.join(Actual_Path, "MDOHypersonic", "Module_Aero", "hyperFolder", "hyper.exe")

    ComString = os.path.join(Actual_Path, "MDOHypersonic", "Module_Aero", "hyperFolder","Analyse_Module_Aero.inp")
    
    process = subprocess.Popen([Hyperpath], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, errors = process.communicate(ComString)
    # Bash show
    # print("Sortie : ", output)
    # print("Erreurs : ", errors)
    return

def GetValues():

    Actual_Path = os.getcwd()
    Actual_Path = Actual_Path.split("MDOHypersonic")[0]
    OutPath = os.path.join(Actual_Path, "MDOHypersonic", "Module_Aero", "hyper.out")

    with open(OutPath, "r") as file:
        data = file.read()

    data = data.split("SOLUTIONS FOR COMPLETE VEHICLE IN WIND AXES")[1]
    data = data.split("COMPUTATIONAL METHODS AND AVERAGE NORMALS")[0]
    line = data.split("\n")
    Data_Line = line[2:]
    Data_Line = Data_Line[:-2]

    Data_Tab = [[float(nombre) for nombre in Line.split("  ") if nombre] for Line in Data_Line]
    CL = [Values[3] for Values in Data_Tab]
    CD = [Values[4] for Values in Data_Tab]
    CM = [Values[5] for Values in Data_Tab]

    return CD,CL,CM
 
def AeroStudie(Avion, Mach, Alphas):
    """ This function aime to do a full Hypersonic aerodynamic analysis"""
    
    FileCreation(Avion,Mach,Alphas)
    RunCalculation()
    CD,CL,CM = GetValues()

    return CD,CL,CM 

if __name__=="__main__":
    Avion = Aircraft.OpenAvion("ICASWT")
    CD,CL,CM = AeroStudie(Avion,3.2,[-2,-1,0,1,2,4,6])
    print(str("CD: " + str(CD) + "\n"))
    print(str("CL: " + str(CL) + "\n"))
    print(str("CM: " + str(CM) + "\n"))