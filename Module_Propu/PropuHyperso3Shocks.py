import numpy as np
import math
import os
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import fsolve

T0 = 288.15         #Sea Level temp in Kelvin
R = 287
Gamma = 1.4
GammaChaud = 1.3
TtCombustion = 1800    #Kelvin
MmAir = 0.0289644      #Molar masses of the air
Rgp = 8.3144621         #Universal cst of perfect gazs
FacteurTechnoCOmbustion = 0.95

class EngineRamJet():

    def __init__(self):
        self.IntakeDiameter = float()
        self.TCombustion = float()
        self.Teta = float()
    
    def OpenEngin(Name):
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

        Engin = EngineRamJet()
        
        parties = {
        "ENGIN": Engin,
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
        
        return Engin
    
    def ISA(Altitude):
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

        return PsInf,Rho, TAltitude



def FlightSpeedFCTMach(Mach,Altitude):

    if Altitude<11000:
        TAltitude = T0-0.0065*Altitude  #Temps at altitude
        Speed_of_Sound = (Gamma*R*TAltitude)**0.5
    else:
        TAltitude = T0-0.0065*11000  #Temps at altitude
        Speed_of_Sound = (Gamma*R*TAltitude)**0.5
    
    FlightSpeed = Mach * Speed_of_Sound
    return FlightSpeed




def Angle1(Mach,Teta,Type="TetaKnow"):
    """Values for the first shock based on:. S. Farokhi, Aircraft propulsion, John Wiley & Sons, 2014."""
    if Type == "Optimize":
        x = [1, 1.5, 2, 2.5, 3, 3.5, 4]
        y = [0, 5.02, 9.69, 13.12, 15.18, 16.19, 16.42]


        f = interp1d(x, y, kind='linear', fill_value="extrapolate")


        Beta = f(Mach)
        Beta = math.radians(Beta)


    if Type == "TetaKnow":
        def equation(Beta1):
            return np.arctan(2 / np.tan(Beta1) * 
                            (Mach**2 * np.sin(Beta1)**2 - 1) / 
                            (Mach**2 * (Gamma + np.cos(2 * Beta1)) + 2)) - Teta

        Beta1_initial_guess = np.radians(30)

        Beta = fsolve(equation, Beta1_initial_guess)[0]
        


    return Beta

def Angle2(Mach,Teta,Type="TetaKnow"):
    """ Values for the second shock based on:. S. Farokhi, Aircraft propulsion, John Wiley & Sons, 2014."""

    x = [1, 1.5, 2, 2.5, 3, 3.5, 4]
    y = [0, 6.42, 11.44, 15.37, 18.45, 20.67, 22.14]

    f = interp1d(x, y, kind='linear', fill_value="extrapolate")

    Beta = f(Mach)
    Beta = math.radians(Beta)
    return Beta

def Deflection(Beta1,Mach1):

    Teta = math.atan(2 / math.tan(Beta1) * (Mach1**2*math.sin(Beta1)**2-1)/(Mach1**2*(Gamma + math.cos(2*Beta1))+2)) 

    return Teta

def Shock(Mach,P,Rho,T,Beta=math.pi/2,Teta=0):
    """Calculate différente values after a shock
    Beta = Ideal angle form the abacus in Rad
    Teta = Ideal deformation from Beta in Rad
    Mach = Mach o fth studied fly condition
    P = Pressure before shock
    Rho = Density before shock"""

    Mn1 = math.sin(Beta)*Mach
    Mn2 = ((1+((Gamma-1)/2)*(Mn1**2))/(Gamma*(Mn1**2)-((Gamma-1)/2)))**0.5
    M2 = Mn2/math.sin(Beta-Teta)

    Rho2 = Rho*(Gamma+1)*Mn1**2/(2+(Gamma-1)*Mn1**2)
    P2 = P*(1+2*Gamma/(Gamma+1)*(Mn1**2-1))
    T2 = P2*Rho*T/(P*Rho2)

    return M2,P2,Rho2,T2

def intake3Shock(RamJet:EngineRamJet,Mach,Altitude):
    """ This function aime to calculer the mach, the pressure and the air density after a 3 shock intake.
    Mach = Mach of the studied fly condition
    Altitude = Altituded of the studied flight condition"""
    P, Rho, T = EngineRamJet.ISA(Altitude)
    Beta1 = Angle1(Mach,RamJet.Teta)
    Beta2 = Beta1+RamJet.Teta
    V = Mach * (Gamma*R*T)**0.5

    M2,P2,Rho2,T2 = Shock(Mach,P,Rho,T,Beta=Beta1,Teta=RamJet.Teta)
    M3,P3,Rho3,T3 = Shock(M2,P2,Rho2,T2,Beta = Beta2-RamJet.Teta)
    M4,P4,Rho4,T4 = Shock(M3,P3,Rho3,T3)

    return M4,P4,Rho4,T4,P,V,Beta1,Beta2

def diffuseur(Mach4,D4,X4,Rho4,T4,P4,Mach5=0.2):
    """COnservation of massique debit the use of bernouille to have Rho and P
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

def CombustionChamberToCol(RamJet:EngineRamJet,M56,D56,P5,M7 = 1):
    """The thuyere is adapater and amorced. Which mean that at section7 mach=1"""

    P6 = P5 * FacteurTechnoCOmbustion
    Rho6 = P5/(RamJet.TCombustion*R)
    V6 = M56*(GammaChaud*R*RamJet.TCombustion)**0.5

    Debit67 = math.pi*((D56/2)**2)*V6*Rho6

    Tt67 = RamJet.TCombustion*(1+(GammaChaud-1)/2*M56)
    T7 = Tt67/(1+(GammaChaud-1)/2*M7)
    V7 = M7*(GammaChaud*R*T7)**0.5
    Rhot67 = Rho6/(((GammaChaud-1)/2*M56**2)**(1/GammaChaud-1))
    Rho7 = Rhot67*(((GammaChaud-1)/2*M7**2)**(1/GammaChaud-1))

    V7 = M7 * (Gamma * R * T7)**0.5
    D7 = 2*((Debit67/(math.pi*V7*Rho7))**0.5)

    Pt67 = P6 * (1 + (GammaChaud - 1) / 2 * M56**2) ** (GammaChaud / (GammaChaud - 1))
    P7 = Pt67 / ((1 + (GammaChaud - 1) / 2 * M7**2) ** (GammaChaud / (GammaChaud - 1)))

    P7bis = P6 + 0.5*Rho6*V6**2-0.5*Rho7*V7**2

    return D7,P7,T7,Rho7

def Section34(RamJet:EngineRamJet,Beta1,Beta2):
    """Just geometrical analysis (see picture)
    IntakeDiameter = Diameter of the front intake
    Beta1 = Angle of the first shock
    Beta2 = Angle of the second shock
    Teta = Angle of the intake defelction
    
    D4 = Diamter after the intake
    X4 = Postion on X of the D4"""

    X2 = math.tan(math.pi/2 - Beta1)*RamJet.IntakeDiameter

    D4 = math.tan(Beta2)*(RamJet.IntakeDiameter/math.sin(RamJet.Teta)-X2)/(1+math.tan(Beta2)/math.sin(RamJet.Teta))
    X4 = (RamJet.IntakeDiameter-D4)/math.sin(RamJet.Teta)
    return D4,X4

def Nozzel(D7,P7,T7,Rho7,Pinf,M7=1):
    Pt78 = P7 * (1 + (GammaChaud - 1) / 2 * M7**2) ** (GammaChaud / (GammaChaud - 1))
    M8 = (((Pt78/Pinf)**((GammaChaud-1)/GammaChaud)-1)*2/(GammaChaud-1))**0.5
    Tt78 = T7*(1+(GammaChaud-1)/2*M7)
    T8 = Tt78/(1+(GammaChaud-1)/2*M8)

    V8 = M8*(GammaChaud*R*T8)**0.5

    return V8

def GraphRamjet(RamJet:EngineRamJet,D4,X4,D5,X5,D6,X6,D7,X7,D8,X8):
    Upper = np.array([0,0])
    Lower = np.array([0,-RamJet.IntakeDiameter])

    Lower = np.vstack([Lower, np.array([X4,-D4])])
    Lower = np.vstack([Lower, np.array([X5,-D5])])
    Lower = np.vstack([Lower, np.array([X6,-D6])])
    Lower = np.vstack([Lower, np.array([X7,-D7])])
    Lower = np.vstack([Lower, np.array([X8,-D8/2-D7/2])])

    Upper = np.vstack([Upper, np.array([Lower[-2,0],0])])
    Upper = np.vstack([Upper, np.array([X8,D8/2-D7/2])])
    plt.figure()
    plt.plot(Lower[:,0],Lower[:,1],color="b")
    plt.plot(Upper[:,0],Upper[:,1],color="b")
    plt.show()

    return


def RamJet(RamJetStudied,Mach, Altitude):
    M4,P4,Rho4,T4,Pinf,Vinf,Beta1,Beta2 = intake3Shock(RamJetStudied,Mach,Altitude)
    D4,X4 = Section34(RamJetStudied,Beta1,Beta2)

    M5,P5,Rho5,T5,D5,X5,DebitMassique = diffuseur(M4,D4,X4,Rho4,T4,P4)

    D7,P7,T7,Rho7 = CombustionChamberToCol(RamJetStudied,M5,D5,P5)
    V8 = Nozzel(D7,P7,T7,Rho7,Pinf)

    F = DebitMassique*(1.1*V8-Vinf)

    # Just random values to draw the RamJet

    D6 = D5
    X6 = X5+D5
    X7 = X6 + D7
    D8 = 1.1*D6
    X8 = X7+D8


    GraphRamjet(RamJetStudied,D4,X4,D5,X5,D6,X6,D7,X7,D8,X8)

    return F


if __name__ == "__main__":

    Mach = 5
    Altitude = 0
    Name = "ICASWT"
    RamJetStudied = EngineRamJet.OpenEngin(Name)
    print(RamJet(RamJetStudied,Mach, Altitude))

    # ram = EngineRamJet()
    # Thrust("ICASWT",5,20000)
