import numpy as np
import math
import os

T0 = 288.15         #Sea Level temp in Kelvin
R = 287
Gamma = 1.4
GammaChaud = 1.3
TtCombustion = 1800    #Kelvin
MmAir = 0.0289644      #Molar masses of the air
Rgp = 8.3144621         #Universal cst of perfect gazs

class EngineRamJet():

    def __init__(self):
        self.Compression_Ratio = float()
        self.Engin_Sampling = float()
        self.Diameter = float()
        self.InternalDiameter = float()
        self.T_Combustion = float()

def FlightSpeedFCTMach(Mach,Altitude):

    if Altitude<11000:
        TAltitude = T0-0.0065*Altitude  #Temps at altitude
        Speed_of_Sound = (Gamma*R*TAltitude)**0.5
    else:
        TAltitude = T0-0.0065*11000  #Temps at altitude
        Speed_of_Sound = (Gamma*R*TAltitude)**0.5
    
    FlightSpeed = Mach * Speed_of_Sound
    return FlightSpeed

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

def ThermoRamjet(RamJet:EngineRamJet,Mach, Altitude,Debit_Inlet):
    H_Base = 11000
    if Altitude<H_Base:
        Pbase = 101325
        Tbase = 288.15
        TAltitude = T0-0.0065*Altitude
        PsInf = Pbase*(TAltitude/Tbase)**(-9.81*MmAir/(Rgp*(-0.0065)))
    else:
        Pbase = 22632.1
        Tbase = 216.65
        PsInf = Pbase*math.exp(-9.81*MmAir*(Altitude-H_Base)/(Rgp*Tbase))

    PtInlet = PsInf * (1 + (Gamma - 1) / 2 * Mach**2) ** (Gamma / (Gamma - 1))
    PtBeforeCombustion = PtInlet * RamJet.Compression_Ratio
    PtAfterCombustion = PtBeforeCombustion*0.98

    TsExit = TtCombustion*(PsInf/PtAfterCombustion)**((GammaChaud-1)/GammaChaud)

    ExitDensity = PsInf/(R*TsExit)

    Vexit = Debit_Inlet*1.1/(ExitDensity*((RamJet.Diameter/2)**2)*math.pi)
    return Vexit,ExitDensity

def Thrust(Engin_name, Mach, Altitude):
    """this function aime to calculte the thrust generate by the studied reactor"""
    Engine_Studied = OpenEngin(Engin_name)

    Flight_Speed = FlightSpeedFCTMach(Mach,Altitude)
    Debit_Inlet = math.pi*((Engine_Studied.Diameter/2)**2)*Flight_Speed
    Speed_Out,ExitDEnsity = ThermoRamjet(Engine_Studied,Mach, Altitude,Debit_Inlet)
    F = ExitDEnsity*Debit_Inlet*(1.1*Speed_Out-Flight_Speed)

    return F

if __name__ == "__main__":

    ram = EngineRamJet()
    Thrust("ICASWT",5,20000)
