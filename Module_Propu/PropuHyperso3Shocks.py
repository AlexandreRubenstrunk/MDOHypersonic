import numpy as np
import math
import os
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import fsolve

# Constants related to atmospheric conditions and gas properties
T0 = 288.15  # Sea Level temperature in Kelvin
R = 287      # Specific gas constant for air [J/(kg*K)]
Gamma = 1.4  # Ratio of specific heats (air)
GammaChaud = 1.3  # Ratio of specific heats (hot gases)
TtCombustion = 1800  # Combustion temperature in Kelvin
MmAir = 0.0289644  # Molar mass of air [kg/mol]
Rgp = 8.3144621  # Universal gas constant [J/(mol*K)]
FacteurTechnoCOmbustion = 0.95  # Efficiency factor for combustion

# Definition of the EngineRamJet class
class EngineRamJet():
    def __init__(self):
        """
        Constructor to initialize the properties of the Ramjet engine.
        """
        self.IntakeDiameter = float()  # Intake diameter [m]
        self.TCombustion = float()  # Combustion temperature [K]
        self.Teta = float()  # Deflection angle [rad]

    @staticmethod
    def OpenEngin(Name):
        """
        Opens and reads the engine configuration from a file,
        and sets up the properties of the Ramjet engine.

        Args:
            Name (str): Name of the engine configuration file.

        Returns:
            EngineRamJet: An instance of the EngineRamJet class with loaded parameters.
        """
        Actual_Path = os.getcwd()
        Actual_Path = Actual_Path.split("MDOHypersonic")[0]  # Adjust the path
        filePath = str(Actual_Path + "MDOHypersonic\\Aircraft\\" + str(Name) + ".txt")

        with open(filePath,'r') as file:
            Data = file.read()
        Parts1Line = Data.split("\n\n")
        Parts = []
        for Part1Line in Parts1Line:
            Part = Part1Line.split("\n")
            Parts.append(Part)

        # Create an instance of the engine
        Engin = EngineRamJet()

        # Dictionary of sections to map parts to the engine object
        parties = {
            "ENGIN": Engin,
        }

        # Load parameters into the engine object
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
        
        return Engin

    @staticmethod
    def ISA(Altitude):
        """
        Calculates the pressure, density, and temperature at a given altitude using the
        International Standard Atmosphere (ISA) model.

        Args:
            Altitude (float): Altitude [m].

        Returns:
            tuple: Pressure [Pa], Density [kg/m³], Temperature [K] at the given altitude.
        """
        H_Base = 11000
        if Altitude < H_Base:
            Pbase = 101325  # Sea level pressure [Pa]
            Tbase = 288.15  # Sea level temperature [K]
            TAltitude = T0 - 0.0065 * Altitude
            PsInf = Pbase * (TAltitude / Tbase) ** (-9.81 * MmAir / (Rgp * (-0.0065)))
        else:
            Pbase = 22632.1  # Stratosphere pressure [Pa]
            TAltitude = 216.65  # Stratosphere temperature [K]
            PsInf = Pbase * math.exp(-9.81 * MmAir * (Altitude - H_Base) / (Rgp * TAltitude))

        Rho = PsInf / (R * TAltitude)

        return PsInf, Rho, TAltitude

def FlightSpeedFCTMach(Mach, Altitude):
    """
    Calculates the flight speed based on Mach number and altitude.

    Args:
        Mach (float): Mach number of the flight.
        Altitude (float): Altitude [m].

    Returns:
        float: Flight speed [m/s].
    """
    if Altitude < 11000:
        TAltitude = T0 - 0.0065 * Altitude  # Temperature at altitude
        Speed_of_Sound = (Gamma * R * TAltitude) ** 0.5
    else:
        TAltitude = T0 - 0.0065 * 11000  # Temperature at 11 km
        Speed_of_Sound = (Gamma * R * TAltitude) ** 0.5

    FlightSpeed = Mach * Speed_of_Sound
    return FlightSpeed

def Angle1(Mach, Teta, Type="TetaKnow"):
    """
    Calculate the angle of the first shock based on the Mach number and the deflection angle.
    Uses either interpolation or a known deflection angle.

    Args:
        Mach (float): Mach number of the flight.
        Teta (float): Deflection angle [rad].
        Type (str): Method to calculate the shock angle ("Optimize" or "TetaKnow").

    Returns:
        float: Shock angle [rad].
    """
    if Type == "Optimize":
        x = [1, 1.5, 2, 2.5, 3, 3.5, 4]
        y = [0, 5.02, 9.69, 13.12, 15.18, 16.19, 16.42]

        f = interp1d(x, y, kind='linear', fill_value="extrapolate")

        Beta = f(Mach)
        Beta = math.radians(Beta)

    if Type == "TetaKnow":
        def equation(Beta1):
            return np.arctan(2 / np.tan(Beta1) * 
                            (Mach ** 2 * np.sin(Beta1) ** 2 - 1) / 
                            (Mach ** 2 * (Gamma + np.cos(2 * Beta1)) + 2)) - Teta

        Beta1_initial_guess = np.radians(30)

        Beta = fsolve(equation, Beta1_initial_guess)[0]

    return Beta

def Angle2(Mach, Teta, Type="TetaKnow"):
    """
    Calculate the angle of the second shock based on the Mach number.

    Args:
        Mach (float): Mach number of the flight.
        Teta (float): Deflection angle [rad].
        Type (str): Method to calculate the shock angle ("TetaKnow").

    Returns:
        float: Shock angle [rad].
    """
    x = [1, 1.5, 2, 2.5, 3, 3.5, 4]
    y = [0, 6.42, 11.44, 15.37, 18.45, 20.67, 22.14]

    f = interp1d(x, y, kind='linear', fill_value="extrapolate")

    Beta = f(Mach)
    Beta = math.radians(Beta)
    return Beta

def Deflection(Beta1, Mach1):
    """
    Calculate the deflection angle given the shock angle and Mach number.

    Args:
        Beta1 (float): Shock angle [rad].
        Mach1 (float): Mach number of the flight.

    Returns:
        float: Deflection angle [rad].
    """
    Teta = math.atan(2 / math.tan(Beta1) * (Mach1 ** 2 * math.sin(Beta1) ** 2 - 1) /
                     (Mach1 ** 2 * (Gamma + math.cos(2 * Beta1)) + 2))
    return Teta

def Shock(Mach, P, Rho, T, Beta=math.pi / 2, Teta=0):
    """
    Calculate the flow properties after a shock.

    Args:
        Mach (float): Mach number before the shock.
        P (float): Pressure before the shock [Pa].
        Rho (float): Density before the shock [kg/m³].
        T (float): Temperature before the shock [K].
        Beta (float): Shock angle [rad].
        Teta (float): Deflection angle [rad].

    Returns:
        tuple: Mach number, Pressure [Pa], Density [kg/m³], Temperature [K] after the shock.
    """
    Mn1 = math.sin(Beta) * Mach
    Mn2 = ((1 + ((Gamma - 1) / 2) * (Mn1 ** 2)) / (Gamma * (Mn1 ** 2) - ((Gamma - 1) / 2))) ** 0.5
    M2 = Mn2 / math.sin(Beta - Teta)

    Rho2 = Rho * (Gamma + 1) * Mn1 ** 2 / (2 + (Gamma - 1) * Mn1 ** 2)
    P2 = P * (1 + 2 * Gamma / (Gamma + 1) * (Mn1 ** 2 - 1))
    T2 = P2 * Rho * T / (P * Rho2)

    return M2, P2, Rho2, T2


import math

def intake3Shock(RamJet: EngineRamJet, Mach, Altitude):
    """
    This function calculates the Mach number, pressure, and air density after a 3-shock intake.
    
    Parameters:
    RamJet (EngineRamJet): An object representing the RamJet engine with various properties like Teta.
    Mach (float): Mach number of the studied flight condition.
    Altitude (float): Altitude of the studied flight condition.
    
    Returns:
    tuple: Mach number, pressure, air density, temperature after shock, initial pressure, velocity, and shock angles.
    """
    # Get the atmospheric values (pressure, density, temperature) for the given altitude
    P, Rho, T = EngineRamJet.ISA(Altitude)
    
    # Calculate the shock angles based on Mach number and intake deflection angle
    Beta1 = Angle1(Mach, RamJet.Teta)
    Beta2 = Beta1 + RamJet.Teta
    
    # Calculate the velocity based on Mach number and temperature
    V = Mach * (Gamma * R * T)**0.5

    # Calculate the conditions after the first, second, and third shocks
    M2, P2, Rho2, T2 = Shock(Mach, P, Rho, T, Beta=Beta1, Teta=RamJet.Teta)
    M3, P3, Rho3, T3 = Shock(M2, P2, Rho2, T2, Beta=Beta2 - RamJet.Teta)
    M4, P4, Rho4, T4 = Shock(M3, P3, Rho3, T3)

    return M4, P4, Rho4, T4, P, V, Beta1, Beta2





def diffuseur(Mach4,D4,X4,Rho4,T4,P4,Mach5=0.2):
    """Conservation of massique debit the use of bernouille to have Rho and P
    WORK IN PROGRESS"""
    V4 = Mach4 * (Gamma * R * T4)**0.5
    Debit4 = math.pi*((D4/2)**2)*V4*Rho4

    Tt45 = T4*(1+(Gamma-1)/2*Mach4**2)
    T5 = Tt45/(1+(Gamma-1)/2*Mach5**2)
    Rhot45 = Rho4/(((Gamma-1)/2*Mach4**2)**(1/Gamma-1))
    Rho5 = Rhot45*(((Gamma-1)/2*Mach5**2)**(1/Gamma-1))

    V5 = Mach5 * (Gamma * R * T5)**0.5
    D5 = 2*((Debit4/(math.pi*V5*Rho5))**0.5)

    Pt45 = P4 * (1 + (Gamma - 1) / 2 * Mach4**2) ** (Gamma / (Gamma - 1))
    P5 = Pt45 / ((1 + (Gamma - 1) / 2 * Mach5**2) ** (Gamma / (Gamma - 1)))

    P5bis = P4 + 0.5*Rho4*V4**2-0.5*Rho5*V5**2

    

    # S4 = math.pi*((D4/2)**2)

    # Th1 = (((Gamma+1)/2)**(-(Gamma+1)/(2*(Gamma-1))))
    # Th2 = (1+(Gamma-1)/2*Mach5**2)**((Gamma+1)/(2*(Gamma-1)))/Mach5

    # S5 = S4/(Th1*Th2)
    # D5bis = 2*(S5/math.pi)**0.5






    # Tt45 = T4*(1+(Gamma-1)/2*Mach4)
    # T5 = Tt45/(1+(Gamma-1)/2*Mach5)
    # V5 = Mach5 * (Gamma * R * T5)**0.5
    # D5 = 2*((Debit4/(math.pi*V5*Rho4))**0.5)
    # P5 = P4 + 0.5*Rho4*V4**2-0.5*Rho4*V5**2

    



    return Mach5,P5,Rho5,T5,D5,D5+X4,Debit4


def CombustionChamberToCol(RamJet: EngineRamJet, M56, D56, P5, M7=1):
    """
    Calculates the conditions in the combustion chamber and nozzle adapter based on the RamJet parameters.
    
    Parameters:
    RamJet (EngineRamJet): An object representing the RamJet engine with combustion temperature.
    M56 (float): Mach number at the combustion chamber inlet.
    D56 (float): Diameter at the combustion chamber inlet.
    P5 (float): Pressure before the combustion chamber.
    M7 (float, optional): Mach number after the combustion chamber. Default is 1 (choked flow).
    
    Returns:
    tuple: Diameter, pressure, temperature, and air density after the combustion chamber.
    """
    # Calculate the pressure, density, and velocity after the combustion chamber
    P6 = P5 * FacteurTechnoCOmbustion
    Rho6 = P5 / (RamJet.TCombustion * R)
    V6 = M56 * (GammaChaud * R * RamJet.TCombustion)**0.5

    Debit67 = math.pi * ((D56 / 2)**2) * V6 * Rho6

    # Calculate the total temperature and pressure after the combustion chamber
    Tt67 = RamJet.TCombustion * (1 + (GammaChaud - 1) / 2 * M56)
    T7 = Tt67 / (1 + (GammaChaud - 1) / 2 * M7)
    
    # Calculate the velocity and density after the combustion chamber
    V7 = M7 * (GammaChaud * R * T7)**0.5
    Rhot67 = Rho6 / (((GammaChaud - 1) / 2 * M56**2)**(1 / GammaChaud - 1))
    Rho7 = Rhot67 * (((GammaChaud - 1) / 2 * M7**2)**(1 / GammaChaud - 1))

    D7 = 2 * ((Debit67 / (math.pi * V7 * Rho7))**0.5)

    Pt67 = P6 * (1 + (GammaChaud - 1) / 2 * M56**2) ** (GammaChaud / (GammaChaud - 1))
    P7 = Pt67 / ((1 + (GammaChaud - 1) / 2 * M7**2) ** (GammaChaud / (GammaChaud - 1)))

    P7bis = P6 + 0.5 * Rho6 * V6**2 - 0.5 * Rho7 * V7**2

    return D7, P7, T7, Rho7


def Section34(RamJet: EngineRamJet, Beta1, Beta2):
    """
    Performs geometrical analysis to determine the diameter and position after the intake.
    
    Parameters:
    RamJet (EngineRamJet): An object representing the RamJet engine.
    Beta1 (float): Angle of the first shock.
    Beta2 (float): Angle of the second shock.
    
    Returns:
    tuple: Diameter after the intake and position after the intake.
    """
    # Calculate the position and diameter after the intake
    X2 = math.tan(math.pi / 2 - Beta1) * RamJet.IntakeDiameter
    D4 = math.tan(Beta2) * (RamJet.IntakeDiameter / math.sin(RamJet.Teta) - X2) / (1 + math.tan(Beta2) / math.sin(RamJet.Teta))
    X4 = (RamJet.IntakeDiameter - D4) / math.sin(RamJet.Teta)
    
    return D4, X4


def Nozzel(D7, P7, T7, Rho7, Pinf, M7=1):
    """
    Calculates the nozzle exit velocity based on the conditions at the nozzle inlet.
    
    Parameters:
    D7 (float): Diameter at the nozzle inlet.
    P7 (float): Pressure at the nozzle inlet.
    T7 (float): Temperature at the nozzle inlet.
    Rho7 (float): Air density at the nozzle inlet.
    Pinf (float): Ambient pressure.
    M7 (float, optional): Mach number at the nozzle exit. Default is 1 (choked flow).
    
    Returns:
    float: Exit velocity at the nozzle.
    """
    # Calculate the total pressure and Mach number at the nozzle exit
    Pt78 = P7 * (1 + (GammaChaud - 1) / 2 * M7**2) ** (GammaChaud / (GammaChaud - 1))
    M8 = (((Pt78 / Pinf)**((GammaChaud - 1) / GammaChaud) - 1) * 2 / (GammaChaud - 1))**0.5

    Tt78 = T7 * (1 + (GammaChaud - 1) / 2 * M7)
    T8 = Tt78 / (1 + (GammaChaud - 1) / 2 * M8)

    # Calculate the exit velocity
    V8 = M8 * (GammaChaud * R * T8)**0.5

    return V8


def GraphRamjet(RamJet: EngineRamJet, D4, X4, D5, X5, D6, X6, D7, X7, D8, X8, Gif=False):
    """
    This function generates a graph of the Ramjet engine's geometry.

    Parameters:
    RamJet (EngineRamJet): The engine instance for the Ramjet being studied.
    D4 (float): Diameter at section 4 of the Ramjet.
    X4 (float): Position of section 4.
    D5 (float): Diameter at section 5 of the Ramjet.
    X5 (float): Position of section 5.
    D6 (float): Diameter at section 6 of the Ramjet.
    X6 (float): Position of section 6.
    D7 (float): Diameter at section 7 of the Ramjet.
    X7 (float): Position of section 7.
    D8 (float): Diameter at section 8 of the Ramjet.
    X8 (float): Position of section 8.
    Gif (bool, optional): If True, the graph will be saved as a GIF. Default is False.

    Returns:
    None
    """
    # Initialize the upper and lower sections of the engine geometry
    Upper = np.array([0, 0])
    Lower = np.array([0, -RamJet.IntakeDiameter])

    # Add successive points to the lower section (ramjet intake and diffuser sections)
    Lower = np.vstack([Lower, np.array([X4, -D4])])
    Lower = np.vstack([Lower, np.array([X5, -D5])])
    Lower = np.vstack([Lower, np.array([X6, -D6])])
    Lower = np.vstack([Lower, np.array([X7, -D7])])
    Lower = np.vstack([Lower, np.array([X8, -D8 / 2 - D7 / 2])])

    # Add successive points to the upper section (ramjet intake and diffuser sections)
    Upper = np.vstack([Upper, np.array([Lower[-2, 0], 0])])
    Upper = np.vstack([Upper, np.array([X8, D8 / 2 - D7 / 2])])

    # Plot the geometry of the Ramjet engine
    plt.figure()
    plt.plot(Lower[:, 0], Lower[:, 1], color="b")  # Plot lower section
    plt.plot(Upper[:, 0], Upper[:, 1], color="b")  # Plot upper section

    # If Gif is True, save the plot as a GIF
    if Gif:
        plt.show(block=False)
        Actual_Path = os.getcwd()
        Actual_Path = Actual_Path.split("MDOHypersonic")[0]
        path = str(Actual_Path + "MDOHypersonic\\Module_Propu\\ImageGIf")

        def get_next_filename(directory, extension=".png"):
            """
            Get the next available filename in the directory with a unique number.

            Parameters:
            directory (str): Directory where the files are stored.
            extension (str, optional): File extension. Default is ".png".

            Returns:
            str: The next available filename with an incremented number.
            """
            # List existing files in the directory
            files = os.listdir(directory)

            # Filter files with the given extension
            numbers = []
            for file in files:
                if file.endswith(extension):
                    try:
                        # Extract the number from the filename (before the extension)
                        number = int(file.replace(extension, ""))
                        numbers.append(number)
                    except ValueError:
                        continue  # Ignore files without a valid number

            # Find the next available number or start from 0 if no files exist
            next_number = max(numbers, default=0) + 1
            return os.path.join(directory, f"{next_number}{extension}")

        name = get_next_filename(path)
        plt.savefig(name, dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot after saving
    else:
        plt.show()  # Just show the plot if Gif is False

    return


def RamJet(RamJetStudied, Mach, Altitude, Gif=False):
    """
    This function simulates the Ramjet engine's performance and generates a thrust value.

    Parameters:
    RamJetStudied (EngineRamJet): The engine instance for the Ramjet being studied.
    Mach (float): The Mach number at the inlet of the Ramjet.
    Altitude (float): The altitude where the engine is operating.
    Gif (bool, optional): If True, a GIF of the Ramjet geometry will be generated. Default is False.

    Returns:
    float: The thrust generated by the Ramjet engine.
    """
    # Call intake3Shock to calculate parameters at the intake
    M4, P4, Rho4, T4, Pinf, Vinf, Beta1, Beta2 = intake3Shock(RamJetStudied, Mach, Altitude)

    # Calculate the diameter and position of section 4
    D4, X4 = Section34(RamJetStudied, Beta1, Beta2)

    # Call the diffuser function to calculate parameters at section 5
    M5, P5, Rho5, T5, D5, X5, DebitMassique = diffuseur(M4, D4, X4, Rho4, T4, P4)

    # Calculate the parameters for the combustion chamber and nozzle
    D7, P7, T7, Rho7 = CombustionChamberToCol(RamJetStudied, M5, D5, P5)
    V8 = Nozzel(D7, P7, T7, Rho7, Pinf)

    # Calculate the thrust using the mass flow rate and velocity differences
    F = DebitMassique * (1.1 * V8 - Vinf)

    # Define random values to simulate the RamJet geometry
    D6 = D5
    X6 = X5 + D5
    X7 = X6 + D7
    D8 = 1.1 * D6
    X8 = X7 + D8

    # Generate the RamJet geometry plot
    if __name__ == "__main__":
        GraphRamjet(RamJetStudied, D4, X4, D5, X5, D6, X6, D7, X7, D8, X8)
    elif Gif:
        GraphRamjet(RamJetStudied, D4, X4, D5, X5, D6, X6, D7, X7, D8, X8, Gif=Gif)

    return F


if __name__ == "__main__":
    # Example usage of the RamJet function
    Mach = 5
    Altitude = 0
    Name = "ICASWT"
    RamJetStudied = EngineRamJet.OpenEngin(Name)
    print(RamJet(RamJetStudied, Mach, Altitude))

    # Uncomment the lines below to test with different parameters
    # ram = EngineRamJet()
    # Thrust("ICASWT", 5, 20000)

