import sys
import os

Actual_Path = os.getcwd()
Actual_Path = Actual_Path.split("MDOHypersonic")[0]
Propu = str(Actual_Path + "MDOHypersonic")
sys.path.insert(1,Propu)

from Module_Propu.PropuHyperso3Shocks import RamJet,EngineRamJet
from Module_Aero.AerodynHyperso import AeroStudie,Wing,Aircraft,Body

Mach = 5
Altitude = 20000
Avion = "ICASWT"

StudyAircarft = Aircraft.OpenAvion(Avion)
StudyRamJet = EngineRamJet.OpenEngin(Avion)

CD,CL,CM = AeroStudie(StudyAircarft,Mach,[-2,-1,0,1,2,4,6])
Thrust = RamJet(StudyRamJet,Mach,Altitude)

print("ok")