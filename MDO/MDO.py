import sys
import os

Actual_Path = os.getcwd()
Actual_Path = Actual_Path.split("MDOHypersonic")[0]
Propu = str(Actual_Path + "MDOHypersonic")
sys.path.insert(1,Propu)

from Module_Propu.PropuHyperso3Shocks import RamJet,EngineRamJet
from Module_Aero.AerodynHyperso import AeroStudie,Wing,Aircraft,Body
from cosapp.base import System
from cosapp.drivers import Optimizer

Mach = 5
Altitude = 20000
Avion = "ICASWT"

StudyAircraft = Aircraft.OpenAvion(Avion)
StudyRamJet = EngineRamJet.OpenEngin(Avion)


class Aero(System):

    def setup(self):

        self.add_inward('Lenght', StudyAircraft.Fuselage.Lenght, unit="m", desc="Lenght of the fuselage")
        self.add_inward('Lenght_Cabine', StudyAircraft.Fuselage.Lenght_Cabine, unit="m", desc="Lenght of the fuselage")
        self.add_inward('Lenght_Nose', StudyAircraft.Fuselage.Lenght_Nose, unit="m", desc="Lenght of the nose")
        self.add_inward('Diameter', StudyAircraft.Fuselage.Diameter, unit="m", desc="Diameter of the fuselage")
        self.add_inward('OffSet_Nose', StudyAircraft.Fuselage.OffSet_Nose, unit="m", desc="Off set of the nose of the fuselage")
        self.add_inward('OffSet_Tail', StudyAircraft.Fuselage.Lenght_Cabine, unit="m", desc="Lenght of the fuselage")

        self.add_inward('Sref', StudyAircraft.Wing.Sref, unit="m**2", desc="Surface of reference")
        self.add_inward('Xref', StudyAircraft.Wing.Xref, unit="m", desc="Chord of reference")
        self.add_inward('AR', StudyAircraft.Wing.AR, desc="Aspect ratio of the wing")
        self.add_inward('TR', StudyAircraft.Wing.TR, desc="Taper ratio")
        self.add_inward('PosX', StudyAircraft.Wing.PosX, unit="m", desc="Position of the wing on X")
        self.add_inward('PosZ', StudyAircraft.Wing.PosZ, unit="m", desc="Position of the wing on Z")
        self.add_inward('Sweep', StudyAircraft.Wing.Sweep, unit="deg", desc="Sweep of the wing")


        self.add_outward('CD', 0.0, desc="Drag Coefficient")
        self.add_outward('CL', 0.0, desc="Lift Coefficient")
        self.add_outward('CM', 0.0, desc="Moment Coefficient")

    def compute(self):
        StudyAircraft = Aircraft.OpenAvion(Avion)

        def update_aircraft_from_aero(aircraft):
            """
            Met à jour les attributs d'un objet `aircraft` avec les données provenant de l'objet `aero`.
            Gère les sous-structures comme `Fuselage` et `Wing`.
            """
            AttWing = list(vars(aircraft.Wing).keys())
            AttWing.remove('AF')

            AttFuse = list(vars(aircraft.Fuselage).keys())
            AttFuse.remove("PosX")
            AttFuse.remove("PosY")
            AttFuse.remove("PosZ")
            AttFuse.remove("Number")

            for i in range(len(AttWing)):
                Value = getattr(self, AttWing[i])
                setattr(aircraft.Wing, AttWing[i], Value)
            
            for j in range(len(AttFuse)):
                Value = getattr(self, AttFuse[j])
                setattr(aircraft.Fuselage, AttFuse[j], Value)

            return aircraft

        StudyAircraft = update_aircraft_from_aero(StudyAircraft)
        CD,CL,CM = AeroStudie(StudyAircraft,Mach,[-2,-1,0,1,2,4,6])
        self.CD = CD[2]
        # return super().compute()




MDO = Aero("MDO")

optim = MDO.add_driver(Optimizer('optim', method='SLSQP'))

min_area = 300
max_area = 400

min_lenght = 50
max_lenght = 70

optim.add_design_var('Sref', lower_bound=min_area, upper_bound=max_area)
optim.add_design_var('Lenght', lower_bound=min_lenght, upper_bound=max_lenght)
optim.set_minimum('CD')

MDO.run_drivers()



# def update_aircraft_from_aero(aero, aircraft):
#     """
#     Met à jour les attributs d'un objet `aircraft` avec les données provenant de l'objet `aero`.
#     Gère les sous-structures comme `Fuselage` et `Wing`.
#     """
#     AttWing = list(vars(aircraft.Wing).keys())
#     AttWing.remove('AF')

#     AttFuse = list(vars(aircraft.Fuselage).keys())

#     for i in range(len(AttWing)):
#         Value = getattr(aero, AttWing[i])
#         setattr(aircraft.Wing, AttWing[i], Value)
    
#     for j in range(len(AttFuse)):
#         Value = getattr(aero, AttFuse[j])
#         setattr(aircraft.Fuselage, AttFuse[j], Value)

    
#     return aircraft

# update_aircraft_from_aero(MDO,StudyAircraft)

# NACELLES
# Lenght		:7.53
# Lenght_Cabine	:3.43
# Lenght_Nose	:2.75
# Diameter	:1.58
# OffSet_Nose	:0
# OffSet_Tail	:0
# PosX		:10.09
# PosY		:0
# PosZ		:1.96




# CD,CL,CM = AeroStudie(StudyAircraft,Mach,[-2,-1,0,1,2,4,6])
# Thrust = RamJet(StudyRamJet,Mach,Altitude)

print("ok")