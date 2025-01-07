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
    """Class representing a body component of the aircraft, such as a fuselage or nacelle."""

    def __init__(self):
        self.Lenght = float()  # Total length of the body
        self.Lenght_Cabine = float()  # Length of the cabin section
        self.Lenght_Nose = float()  # Length of the nose section
        self.Diameter = float()  # Diameter of the body
        self.OffSet_Nose = float()  # Vertical offset of the nose
        self.OffSet_Tail = float()  # Vertical offset of the tail
        self.PosX = float(0)  # X-position of the body
        self.PosY = float(0)  # Y-position of the body
        self.PosZ = float(0)  # Z-position of the body
        self.Number = float(1)  # Identifier for the body

class Wing:
    """Class representing the wing of the aircraft."""

    def __init__(self):
        self.Sref = float()  # Reference area of the wing
        self.Xref = float()  # X-reference point of the wing
        self.AR = float()  # Aspect ratio of the wing
        self.TR = float()  # Taper ratio of the wing
        self.AF = str()  # Airfoil type
        self.PosX = float()  # X-position of the wing
        self.PosZ = float()  # Z-position of the wing
        self.Sweep = float()  # Sweep angle of the wing (in degrees)

class Aircraft:
    """Class representing an aircraft, including its components."""

    def __init__(self):
        self.Name = str()  # Name of the aircraft
        self.Fuselage = Body()  # Fuselage component
        self.Wing = Wing()  # Wing component
        self.Nacelles = Body()  # Nacelles component

    @staticmethod
    def OpenAvion(Name):
        """
        Opens a TXT file representing an aircraft and loads its attributes.

        Parameters:
            Name (str): Name of the aircraft file (without extension).

        Returns:
            Aircraft: An Aircraft object with its values populated.
        """
        Actual_Path = os.getcwd()
        Actual_Path = Actual_Path.split("MDOHypersonic")[0]
        filePath = str(Actual_Path + "MDOHypersonic\\Aircraft\\" + str(Name) + ".txt")

        with open(filePath, 'r') as file:
            Data = file.read()

        Parts1Line = Data.split("\n\n")
        Parts = [part.split("\n") for part in Parts1Line]

        Plane = Aircraft()

        parties = {
            "FUSELAGE": Plane.Fuselage,
            "WING": Plane.Wing,
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

def PointBody(Body: Body, NB_CF, NB_Point_CF):
    """
    Defines the characteristic points of the fuselage.

    Parameters:
        Body (Body): The body (e.g., fuselage) to define points for.
        NB_CF (int): Number of contours.
        NB_Point_CF (int): Number of points per contour.

    Returns:
        numpy.ndarray: A 3D array containing the points, sorted by contour and point number.
    """
    Point = np.zeros((NB_CF, NB_Point_CF, 3))

    if NB_CF >= 4:
        Additionnal = (NB_CF - 4)
        Disciminant = Additionnal % 3

        Number_Controur_Per_Part = Additionnal // 3

        if Disciminant == 0:
            Front_Cabine_Indexe = Number_Controur_Per_Part + 1
            Rear_Cabin_Indexe = Front_Cabine_Indexe + Number_Controur_Per_Part + 1
            Rear_Fuselage_Indexe = Rear_Cabin_Indexe + Number_Controur_Per_Part + 1
        elif Disciminant == 1:
            Front_Cabine_Indexe = Number_Controur_Per_Part + 1
            Rear_Cabin_Indexe = Front_Cabine_Indexe + Number_Controur_Per_Part + 2
            Rear_Fuselage_Indexe = Rear_Cabin_Indexe + Number_Controur_Per_Part + 1
        else:
            Front_Cabine_Indexe = Number_Controur_Per_Part + 2
            Rear_Cabin_Indexe = Front_Cabine_Indexe + Number_Controur_Per_Part + 2
            Rear_Fuselage_Indexe = Rear_Cabin_Indexe + Number_Controur_Per_Part + 1

        Point[0, :, 0] = Body.PosX
        Point[0, :, 2] = Body.OffSet_Nose + Body.PosZ

        Point[Rear_Fuselage_Indexe, :, 2] = Body.OffSet_Tail + Body.PosZ
        Point[Rear_Fuselage_Indexe, :, 0] = Body.Lenght + Body.PosX

        Point[Front_Cabine_Indexe, :, 0] = Body.Lenght_Nose + Body.PosX
        Point[Rear_Cabin_Indexe, :, 0] = Body.Lenght_Nose + Body.Lenght_Cabine + Body.PosX

        Teta = math.pi / (NB_Point_CF - 1)

        for i in range(NB_Point_CF):
            Point[Front_Cabine_Indexe, i, 1] = (Body.Diameter * math.sin(i * Teta) / 2) + Body.PosY
            Point[Rear_Cabin_Indexe, i, 1] = (Body.Diameter * math.sin(i * Teta) / 2) + Body.PosY

            Point[Front_Cabine_Indexe, i, 2] = (Body.Diameter * math.cos(i * Teta) / 2) + Body.PosZ
            Point[Rear_Cabin_Indexe, i, 2] = (Body.Diameter * math.cos(i * Teta) / 2) + Body.PosZ

            for j in range(Front_Cabine_Indexe):
                if Front_Cabine_Indexe > 1:
                    Indexe_Norm = (j / (Front_Cabine_Indexe - 1))
                    Point[j, i, :] = (Point[0, i, :] * (1 - Indexe_Norm) + Point[Front_Cabine_Indexe, i, :] * Indexe_Norm)

            for j in range(Front_Cabine_Indexe + 1, Rear_Cabin_Indexe):
                if Rear_Cabin_Indexe - Front_Cabine_Indexe > 1:
                    Indexe_Norm = ((j - Front_Cabine_Indexe) / ((Rear_Cabin_Indexe - Front_Cabine_Indexe) - 1))
                    Point[j, i, :] = (Point[Front_Cabine_Indexe, i, :] * (1 - Indexe_Norm) + Point[Rear_Cabin_Indexe, i, :] * Indexe_Norm)

            for j in range(Rear_Cabin_Indexe + 1, Rear_Fuselage_Indexe):
                if Rear_Fuselage_Indexe - Rear_Cabin_Indexe > 1:
                    Indexe_Norm = ((j - Rear_Cabin_Indexe) / ((Rear_Fuselage_Indexe - Rear_Cabin_Indexe) - 1))
                    Point[j, i, :] = (Point[Rear_Cabin_Indexe, i, :] * (1 - Indexe_Norm) + Point[Rear_Fuselage_Indexe, i, :] * Indexe_Norm)

    return Point

def Point2String(PT, STR, NB_C, NB_Point_C):
    """
    Converts a 3D matrix of points into a formatted string representation.

    Parameters:
    - PT: A 3D matrix containing the points to convert.
    - STR: The initial string to append the points to.
    - NB_C: Number of contours.
    - NB_Point_C: Number of points per contour.

    Returns:
    - STR: The updated string with the points appended.
    """
    max_longueur = 79  # Maximum line length for the formatted string.

    for i in range(NB_C):  # Iterate through each contour.
        STR += "\n"  # Start a new line for each contour.
        for j in range(NB_Point_C):  # Iterate through each point in the contour.
            # Format the point's coordinates as scientific notation.
            point_str = f"{PT[i, j, 0]:.8E} {PT[i, j, 1]:.8E} {PT[i, j, 2]:.8E}"

            # Check if adding the new point would exceed the line length limit.
            if len(STR) > 0 and len(STR) + len(point_str) + 1 > max_longueur:
                STR += "\n"  # Start a new line if the limit is exceeded.

            STR += point_str + " "  # Append the point and a space.
    return STR


def CreationFuselage(Avion: Aircraft, NB_CF: int, NB_Point_CF: int):
    """
    Creates the string definition of the fuselage based on the aircraft data.

    Parameters:
    - Avion: The aircraft object containing fuselage data.
    - NB_CF: Number of contours for the fuselage.
    - NB_Point_CF: Number of points per contour.

    Returns:
    - STR: The formatted fuselage definition string.
    - PF: The array of points defining the fuselage.
    """
    STR = str("'Body'\n1\t" + str(NB_CF) + "\t" + str(NB_Point_CF) +
              "\t0\t0 0 0\t0 0 0\t1 1 1\t1")

    # Generate the points defining the fuselage.
    PF = PointBody(Avion.Fuselage, NB_CF, NB_Point_CF)

    # Convert the points to a string and append them to the fuselage definition.
    STR = Point2String(PF, STR, NB_CF, NB_Point_CF)

    return STR, PF

def OpenAF(Avion):
    """
    Reads the airfoil profile data from a file.

    Parameters:
    - Avion: The aircraft object containing airfoil profile information.

    Returns:
    - AFcoordinat: A numpy array of the airfoil coordinates.
    """
    Actual_Path = os.getcwd()  # Get the current working directory.
    Actual_Path = Actual_Path.split("MDOHypersonic")[0]  # Extract the base path.
    # Construct the full path to the airfoil profile file.
    AFPath = str(Actual_Path + "MDOHypersonic\\Aircraft\\Profile\\" + Avion.Wing.AF + ".txt")

    # Read the airfoil profile data from the file.
    with open(AFPath, 'r') as file:
        DataAF = file.read()
    DataAF = DataAF.split("\n")  # Split the data into lines.
    # Convert the data into a numpy array of floats.
    AFcoordinat = np.array([list(map(float, item.split('\t'))) for item in DataAF])
    return AFcoordinat


def CreationPointWing(Avion: Aircraft, NB_CW):
    """
    Generates the 3D coordinates of the wing's upper and lower surfaces.

    Parameters:
    - Avion: The aircraft object containing wing geometry.
    - NB_CW: Number of wing sections (chordwise resolution).

    Returns:
    - PointWing_Upper: 3D coordinates of the wing's upper surface.
    - PointWing_Lower: 3D coordinates of the wing's lower surface.
    - NB_Point_CW: Number of points per chordwise section.
    """
    AFCoordinat = OpenAF(Avion)  # Open the airfoil profile data.
    PointWing_Upper = np.zeros((NB_CW, AFCoordinat.shape[0], 3))  # Initialize the upper surface points.
    PointWing_Lower = np.zeros((NB_CW, AFCoordinat.shape[0], 3))  # Initialize the lower surface points.

    # Separate the airfoil coordinates into x and y components.
    Dim_0 = AFCoordinat[:, 0]
    Dim_1 = AFCoordinat[:, 1]

    # Create a linear interpolation function for the airfoil profile.
    InterAF = interp1d(Dim_0, Dim_1, kind='linear', fill_value="extrapolate")

    # Calculate the wing span.
    Span = (Avion.Wing.Sref * Avion.Wing.AR) ** 0.5

    for i in range(NB_CW):
        # Calculate the y-position using sinusoidal distribution.
        y_position = (Span / 2) * np.sin(i * np.pi / (2 * (NB_CW - 1)))

        # Assign the y-position to the upper and lower surfaces.
        PointWing_Upper[i, :, 1] = y_position
        PointWing_Lower[i, :, 1] = y_position

        # Calculate wing geometry.
        MAC = Avion.Wing.Sref / Span
        Croot = 2 * MAC / (1 + Avion.Wing.TR)

        for j in range(AFCoordinat.shape[0]):
            # Calculate the x and z positions for the upper and lower surfaces.
            PosSweep = PointWing_Lower[i, j, 1] * math.tan(math.radians(Avion.Wing.Sweep))
            PointWing_Upper[i, j, 0] = Avion.Wing.PosX + j * Croot * (
                1 - (1 - Avion.Wing.TR) * 2 * PointWing_Upper[i, j, 1] / Span) / (AFCoordinat.shape[0] - 1)
            PointWing_Upper[i, j, 0] += PosSweep
            PointWing_Upper[i, j, 2] = Avion.Wing.PosZ + Croot * (
                1 - (1 - Avion.Wing.TR) * 2 * PointWing_Upper[i, j, 1] / Span) * InterAF(j / (AFCoordinat.shape[0] - 1))
            
            PointWing_Lower[i, j, 0] = Avion.Wing.PosX + j * Croot * (
                1 - (1 - Avion.Wing.TR) * 2 * PointWing_Upper[i, j, 1] / Span) / (AFCoordinat.shape[0] - 1)
            PointWing_Lower[i, j, 0] += PosSweep
            PointWing_Lower[i, j, 2] = Avion.Wing.PosZ - Croot * (
                1 - (1 - Avion.Wing.TR) * 2 * PointWing_Upper[i, j, 1] / Span) * InterAF(j / (AFCoordinat.shape[0] - 1))

    return PointWing_Upper, PointWing_Lower, AFCoordinat.shape[0]

def CreationWing(Avion, NB_CW):
    """
    Creates the string representation and 3D points of the wing.

    Parameters:
    - Avion: The aircraft object containing wing data.
    - NB_CW: Number of wing sections (chordwise resolution).

    Returns:
    - STR_Wing: The formatted wing definition string (upper and lower surfaces).
    - PW: The combined 3D points of the wing (upper and lower surfaces).
    """
    # Generate 3D coordinates for the upper and lower surfaces of the wing.
    PW_Upper, PW_Lower, NB_Point_CW = CreationPointWing(Avion, NB_CW)

    # Create the string definition for the upper surface.
    STR_Upper = str("\n'WING-UPPER'\n1\t" + str(NB_CW) + "\t" + str(NB_Point_CW) +
                    "\t0\t0 0 0\t0 0 0\t1 1 1\t1")
    STR_Upper = Point2String(PW_Upper, STR_Upper, NB_CW, NB_Point_CW)

    # Create the string definition for the lower surface.
    STR_Lower = str("\n'WING-LOWER'\n1\t" + str(NB_CW) + "\t" + str(NB_Point_CW) +
                    "\t0\t0 0 0\t0 0 0\t1 1 1\t1")
    STR_Lower = Point2String(PW_Lower, STR_Lower, NB_CW, NB_Point_CW)

    # Combine the upper and lower strings and points.
    STR_Wing = STR_Upper + STR_Lower
    PW = np.concatenate((PW_Upper, PW_Lower))  # Combine the point arrays.

    return STR_Wing, PW


def CreationNacelles(Avion: Aircraft, NB_CN, NB_Point_CN, Number):
    """
    Creates the string representation and 3D points of the nacelles.

    Parameters:
    - Avion: The aircraft object containing nacelle data.
    - NB_CN: Number of nacelle contours.
    - NB_Point_CN: Number of points per nacelle contour.
    - Number: The identifier for the nacelle (used in the string).

    Returns:
    - STR: The formatted nacelle definition string.
    - PN: The 3D points defining the nacelle geometry.
    """
    STR = str("\n'BODY'\n" + str(Number) + "\t" + str(NB_CN) + "\t" +
              str(NB_Point_CN) + "\t0\t0 0 0\t0 0 0\t1 1 1\t1")

    # Generate the points defining the nacelle geometry.
    PN = PointBody(Avion.Nacelles, NB_CN, NB_Point_CN)

    # Convert the points to a string and append them to the nacelle definition.
    STR = Point2String(PN, STR, NB_CN, NB_Point_CN)

    return STR, PN


def CFDFileWing(Avion: Aircraft, WingPoint):
    """
    Generates a CFD input file containing the wing geometry.

    Parameters:
    - Avion: The aircraft object containing wing data.
    - WingPoint: A 3D array of the wing's points (upper and lower surfaces).

    Returns:
    - None: Writes the CFD file directly to disk.
    """
    # Get the base path of the project directory.
    Actual_Path = os.getcwd()
    Actual_Path = Actual_Path.split("MDOHypersonic")[0]
    # Construct the file path for the CFD wing file.
    AFPath = str(Actual_Path + "MDOHypersonic\\Aircraft\\CFDFile\\" + Avion.Name + "CFDWing.txt")

    # Split the WingPoint array into upper and lower surfaces.
    midpoint = WingPoint.shape[0] // 2
    Upper = WingPoint[:midpoint, :, :]
    Lower = WingPoint[midpoint:, :, :]

    # Create the root and tip points by concatenating upper and reversed lower surfaces.
    Root = np.concatenate((Upper[0, :, :], Lower[0, :, :][::-1]))
    Tip = np.concatenate((Upper[-1, :, :], Lower[-1, :, :][::-1]))

    # Combine the root and tip points.
    CfdFilePoint = np.concatenate((Root, Tip))

    # Initialize an empty string to store the file content.
    STR = str()
    # Iterate through the points and format them as lines in the CFD file.
    for i in range(CfdFilePoint.shape[0]):
        STR += str("1\t" + str(i + 1) + "\t" +
                   " ".join(map(str, CfdFilePoint[i])) + "\n")

    # Write the formatted string to the CFD file.
    with open(AFPath, "w") as File:
        File.write(STR)

    

def FileCreation(Avion, Mach, Alphas, NB_CF=4, NB_Point_CF=5, NB_CW=6, CFD=False, Gif=False):
    """
    Creates the necessary files to compute an aircraft analysis with Hyper.

    Parameters:
    - Avion: The aircraft object containing its geometry and attributes.
    - Mach: The Mach number for the analysis.
    - Alphas: A list of angles of attack (AoA) to be studied.
    - NB_CF: Number of fuselage/nacelle contours (default: 4).
    - NB_Point_CF: Number of points per contour (default: 5).
    - NB_CW: Number of wing sections (default: 6).
    - CFD: If True, generates a CFD file for the wing.
    - Gif: If True, saves visualization frames for creating a GIF.

    Returns:
    - None
    """
    # Initialize the file content with a descriptive title.
    STRTotal = "Analyse module aero\n"

    # Create fuselage string and points.
    FuselageSTR, FuselagePoint = CreationFuselage(Avion, NB_CF, NB_Point_CF)
    # Create wing string and points.
    WingSTR, WingPoint = CreationWing(Avion, NB_CW)
    # Create nacelle string and points (assuming identifier 2 for the nacelles).
    NacellesSTR, NacellesPoint = CreationNacelles(Avion, NB_CF, NB_Point_CF, 2)

    # Generate a CFD file for the wing if CFD is enabled.
    if CFD:
        CFDFileWing(Avion, WingPoint)

    # Plot the aircraft visualization (either for GIF or static display).
    if Gif:
        PlotAircraft(FuselagePoint, WingPoint, NacellesPoint, Gif=Gif)
    elif __name__ == "__main__":
        PlotAircraft(FuselagePoint, WingPoint, NacellesPoint)
        PlotAircraftMesh(FuselagePoint, WingPoint, NacellesPoint)

    # Combine all component strings into the full geometry description.
    STRTotal += FuselageSTR
    STRTotal += WingSTR
    STRTotal += NacellesSTR

    # Define the file path for the geometry file.
    Actual_Path = os.getcwd()
    Actual_Path = Actual_Path.split("MDOHypersonic")[0]
    FilePath = str(Actual_Path + "MDOHypersonic\\Module_Aero\\hyperFolder\\Analyse_Module_Aero.wgs")

    # Write the geometry description to the file.
    with open(FilePath, "w") as File:
        File.write(STRTotal)

    # Calculate the span and mean aerodynamic chord (MAC) of the wing.
    Span = (Avion.Wing.Sref * Avion.Wing.AR) ** 0.5
    MAC = Avion.Wing.Sref / Span

    # Create the input file content for the analysis.
    STRinp = (
        f"&hyp  title='Analyse module aero',\n"
        f"wgsFileName= 'hyperFolder\\Analyse_Module_Aero.wgs' ,\n"
        f"cmethods=7,1,1,7,\nemethods=5,5,5,5,\nmach={Mach}, sref={Avion.Wing.Sref}, cbar={MAC},\n"
        f"alpha={','.join(map(str, Alphas))},\n"
        f"xref={Avion.Wing.Xref}, span={Span}/\n"
    )

    # Define the file path for the input file.
    INPFilePath = str(Actual_Path + "MDOHypersonic\\Module_Aero\\hyperFolder\\Analyse_Module_Aero.inp")

    # Write the input file content to the file.
    with open(INPFilePath, "w") as File:
        File.write(STRinp)

    return


def PlotAircraftMesh(PointFuselage, Pointwing, PointNacelles):
    """
    Plots all points of the aircraft with symmetry along the Y-axis.

    Parameters:
    - PointFuselage: Numpy array of fuselage points (3D coordinates).
    - Pointwing: Numpy array of wing points (3D coordinates).
    - PointNacelles: Numpy array of nacelle points (3D coordinates).
    """
    # Create a 3D plot figure.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def plot_symmetric_points(points, color, label):
        """
        Helper function to plot points with symmetry along the Y-axis.

        Parameters:
        - points: Numpy array of 3D coordinates.
        - color: Color of the points.
        - label: Label for the original side.
        """
        x = points[:, :, 0].flatten()
        y = points[:, :, 1].flatten()
        z = points[:, :, 2].flatten()

        # Plot original points.
        ax.scatter(x, y, z, label=label, color=color)
        # Plot mirrored points.
        ax.scatter(x, -y, z, color=color)

    # Plot each component with its respective color and label.
    plot_symmetric_points(PointFuselage, 'b', 'Fuselage')
    plot_symmetric_points(Pointwing, 'r', 'Wing')
    plot_symmetric_points(PointNacelles, 'g', 'Nacelles')

    # Set axis labels and legend.
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.axis("equal")
    plt.show()


def PlotAircraft(PointFuselage, Pointwing, PointNacelles, Gif=False):
    """
    Plots the aircraft as solid surfaces with symmetry along the Y-axis.

    Parameters:
    - PointFuselage, Pointwing, PointNacelles: Numpy arrays containing the 3D points.
    - Gif: If True, saves the plot as a frame for GIF generation.
    """
    # Create a 3D plot figure.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def plot_symmetric_cylinder(points, color):
        """Plots cylindrical surfaces with symmetry along the Y-axis."""
        x = points[:, :, 0]
        y = points[:, :, 1]
        z = points[:, :, 2]
        ax.plot_surface(x, y, z, color=color, alpha=0.7, shade=True)
        ax.plot_surface(x, -y, z, color=color, alpha=0.7, shade=True)

    def plot_symmetric_surface(points, color):
        """Plots triangulated surfaces with symmetry along the Y-axis."""
        x = points[:, :, 0].flatten()
        y = points[:, :, 1].flatten()
        z = points[:, :, 2].flatten()
        tri = Triangulation(x, y)
        ax.plot_trisurf(x, y, z, triangles=tri.triangles, color=color, alpha=0.7, shade=True)
        ax.plot_trisurf(x, -y, z, triangles=tri.triangles, color=color, alpha=0.7, shade=True)

    # Plot fuselage and nacelles as cylinders and wing as surfaces.
    plot_symmetric_cylinder(PointFuselage, 'b')
    plot_symmetric_cylinder(PointNacelles, 'g')
    plot_symmetric_surface(Pointwing, 'r')

    # Add legend manually.
    proxy_fuselage = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, label='Fuselage')
    proxy_wing = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Wing')
    proxy_nacelles = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Nacelles')
    ax.legend(handles=[proxy_fuselage, proxy_wing, proxy_nacelles])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.axis("equal")

    if Gif:
        # Save the plot as a frame for the GIF.
        Actual_Path = os.getcwd()
        Actual_Path = Actual_Path.split("MDOHypersonic")[0]
        path = str(Actual_Path + "MDOHypersonic\\Module_Aero\\ImageGIf")

        def get_next_filename(directory, extension=".png"):
            """Returns the next unique filename with a numeric suffix in the given directory."""
            # Extract numeric suffixes from existing files with the specified extension
            numbers = [
                int(file.removesuffix(extension))
                for file in os.listdir(directory)
                if file.endswith(extension) and file.removesuffix(extension).isdigit()
            ]
            
            # Determine the next available number
            next_number = max(numbers, default=0) + 1
            
            # Return the full path for the next file
            return os.path.join(directory, f"{next_number}{extension}")
        name = get_next_filename(path)
        plt.savefig(name, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def RunCalculation():
    """
    This function sets up the path for the 'hyper.exe' executable and 
    runs the corresponding calculation using the subprocess module.
    It communicates the necessary input file to the executable and retrieves
    any output or errors produced during execution.
    """
    
    # Get the current working directory
    Actual_Path = os.getcwd()
    
    # Navigate to the root directory of the project by removing 'MDOHypersonic' from the path
    Actual_Path = Actual_Path.split("MDOHypersonic")[0]
    
    # Construct the path to the 'hyper.exe' executable
    Hyperpath = os.path.join(Actual_Path, "MDOHypersonic", "Module_Aero", "hyperFolder", "hyper.exe")
    
    # Construct the command string to run with the executable
    ComString = os.path.join(Actual_Path, "MDOHypersonic", "Module_Aero", "hyperFolder", "Analyse_Module_Aero.inp")
    
    # Use subprocess to start the process, send the input file to 'hyper.exe', and capture the output and errors
    process = subprocess.Popen([Hyperpath], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Communicate the input and get the output and error messages
    output, errors = process.communicate(ComString)
    
    # Uncomment the following lines to print the output and errors for debugging
    # print("Output: ", output)
    # print("Errors: ", errors)
    
    return  # No return value

def GetValues():
    """
    This function reads the output file generated by the calculation ('hyper.out'),
    parses the data for aerodynamic coefficients (CD, CL, CM), and returns them as lists.
    """
    
    # Get the current working directory
    Actual_Path = os.getcwd()
    
    # Construct the path to the output file
    OutPath = os.path.join(Actual_Path, "hyper.out")
    
    # Open the output file and read its content
    with open(OutPath, "r") as file:
        data = file.read()
    
    # Parse the data to extract the relevant portion between specific markers
    data = data.split("SOLUTIONS FOR COMPLETE VEHICLE IN WIND AXES")[1]
    data = data.split("COMPUTATIONAL METHODS AND AVERAGE NORMALS")[0]
    
    # Split the data into lines and extract the data lines
    line = data.split("\n")
    Data_Line = line[2:]  # Skip the first two lines of the data
    Data_Line = Data_Line[:-2]  # Skip the last two lines of the data
    
    # Convert the data into a list of floats
    Data_Tab = [[float(nombre) for nombre in Line.split("  ") if nombre] for Line in Data_Line]
    
    # Extract the aerodynamic coefficients (CD, CL, CM) from the data
    CL = [Values[3] for Values in Data_Tab]
    CD = [Values[4] for Values in Data_Tab]
    CM = [Values[5] for Values in Data_Tab]

    return CD, CL, CM  # Return the lists of aerodynamic coefficients

def AeroStudie(Avion, Mach, Alphas, Gif=False):
    """
    This function performs a complete hypersonic aerodynamic analysis for a given aircraft (Avion),
    Mach number, and range of angles of attack (Alphas). It first creates necessary input files,
    runs the calculation, and retrieves the aerodynamic coefficients (CD, CL, CM).
    
    Parameters:
    Avion (Aircraft object): The aircraft to be analyzed
    Mach (float): The Mach number for the analysis
    Alphas (list of floats): The list of angles of attack to be considered
    Gif (bool, optional): Whether to generate a GIF (default is False)
    
    Returns:
    tuple: A tuple containing lists of CD, CL, and CM coefficients
    """
    
    # Create the input files for the analysis
    FileCreation(Avion, Mach, Alphas, Gif=Gif)
    
    # Run the calculation using the previously defined function
    RunCalculation()
    
    # Retrieve the aerodynamic coefficients
    CD, CL, CM = GetValues()

    return CD, CL, CM  # Return the aerodynamic coefficients

# Main execution block
if __name__ == "__main__":
    # Open the aircraft object (using a method specific to your project)
    Avion = Aircraft.OpenAvion("ProtoConcord")
    
    # Perform the aerodynamic study for the given aircraft, Mach number, and angles of attack
    CD, CL, CM = AeroStudie(Avion, 3.2, [-2, -1, 0, 1, 2, 4, 6])
    
    # Print the results
    print(str("CD: " + str(CD) + "\n"))
    print(str("CL: " + str(CL) + "\n"))
    print(str("CM: " + str(CM) + "\n"))
