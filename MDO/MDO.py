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

gamma = 1.4
R = 287
Mach = 5
Altitude = 20000
Avion = "ProtoConcord"
MTOW = 185000

StudyAircraft = Aircraft.OpenAvion(Avion)
StudyRamJet = EngineRamJet.OpenEngin(Avion)
PsInf,Rho, TAltitude = EngineRamJet.ISA(Altitude)
V = Mach*(gamma*R*TAltitude)**0.5
count=0

class Propu(System):
    def setup(self):

        self.add_inward('IntakeDiameter', StudyRamJet.IntakeDiameter, unit="m", desc="diameter at the intake of the ramjet")
        self.add_inward('TCombustion',StudyRamJet.TCombustion,unit="K",desc="Temperature inside the combustion chamber")
        self.add_inward('Teta',StudyRamJet.Teta,unit="rad",desc="Value of the angle of the intake rampe")
        self.add_inward('Number',1,desc='Number of ramjet')
        
        self.add_inward("Drag",0,unit="N",desc="Represente the value of the aerodynamic drag")

        self.add_outward('Thrust',0,unit="N",desc="Thrust generate by the propulsion systeme")

        

    def compute(self):
        StudiedRamJet = EngineRamJet.OpenEngin(Avion)
        def update_ramjet_from_propu(Ramjet):
            """
            Met à jour les attributs d'un objet `aircraft` avec les données provenant de l'objet `aero`.
            Gère les sous-structures comme `Fuselage` et `Wing`.
            """
            Att = list(vars(Ramjet).keys())

            for i in range(len(Att)):
                Value = getattr(self, Att[i])
                setattr(RamJet, Att[i], Value)

            return RamJet
        
        if self.IntakeDiameter<0:
            print("ok")
        StudiedRamJet = update_ramjet_from_propu(StudiedRamJet)
        print(self.Thrust)

        self.Thrust = self.Number*RamJet(StudiedRamJet,Mach, Altitude)

        

class Aero(System):

    def setup(self):

        self.add_inward('Lenght', StudyAircraft.Fuselage.Lenght, unit="m", desc="Lenght of the fuselage")
        self.add_inward('Lenght_Cabine', StudyAircraft.Fuselage.Lenght_Cabine, unit="m", desc="Lenght of the fuselage")
        self.add_inward('Lenght_Nose', StudyAircraft.Fuselage.Lenght_Nose, unit="m", desc="Lenght of the nose")
        self.add_inward('Diameter', StudyAircraft.Fuselage.Diameter, unit="m", desc="Diameter of the fuselage")
        self.add_inward('OffSet_Nose', StudyAircraft.Fuselage.OffSet_Nose, unit="m", desc="Off set of the nose of the fuselage")
        self.add_inward('OffSet_Tail', StudyAircraft.Fuselage.OffSet_Tail, unit="m", desc="Lenght of the fuselage")

        self.add_inward('Sref', StudyAircraft.Wing.Sref, unit="m**2", desc="Surface of reference")
        self.add_inward('Xref', StudyAircraft.Wing.Xref, unit="m", desc="Chord of reference")
        self.add_inward('AR', StudyAircraft.Wing.AR, desc="Aspect ratio of the wing")
        self.add_inward('TR', StudyAircraft.Wing.TR, desc="Taper ratio")
        self.add_inward('PosX', StudyAircraft.Wing.PosX, unit="m", desc="Position of the wing on X")
        self.add_inward('PosZ', StudyAircraft.Wing.PosZ, unit="m", desc="Position of the wing on Z")
        self.add_inward('Sweep', StudyAircraft.Wing.Sweep, unit="deg", desc="Sweep of the wing")

        self.add_inward('Rho',Rho,unit="kg/m**3",desc="Air density at the studied altitude")
        self.add_inward('V',V,unit='m/s',desc="Speed of the studied flight")
        self.add_inward('MTOW',MTOW,unit='kg',desc="Maximum Take Off Weight")
        self.add_inward('count',count,desc="Count iteration")
        self.add_inward('AoA',5,unit="deg",desc="Angle of attack")


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
        print(self.CL)
        print(self.CD)
        print(self.AoA)
        print("\n")
        self.count= self.count+1
        StudyAircraft = update_aircraft_from_aero(StudyAircraft)
        CD,CL,CM = AeroStudie(StudyAircraft,Mach,[self.AoA])
        self.CD = CD[0]
        self.CL = CL[0]
        self.CM = CM[0]
        return super().compute()




MDO = Aero("MDO")

optim = MDO.add_driver(Optimizer('optim', method='COBYLA'))

min_area = 300
max_area = 600

min_lenght = 50
max_lenght = 70

optim.options['tol'] = 1e-6
optim.add_unknown('Sref', lower_bound=min_area, upper_bound=max_area)
optim.add_unknown('Lenght', lower_bound=min_lenght, upper_bound=max_lenght)
optim.add_unknown('AoA', lower_bound=0, upper_bound=5)
optim.add_unknown(["Sref",'Xref', 'AR', 'TR', 'PosX', 'PosZ', 'Sweep'])
# optim.add_constraints([

#     'CL>0',
#     '0.5*Sref*CL*Rho*V*V>=MTOW*9.81'

# ])

optim.set_minimum('CD')

MDO.run_drivers()

Drag = 0.5*MDO.Sref*MDO.CD*MDO.Rho*MDO.V*MDO.V

# MDOEngin = Propu("MDOEngin")
# MDOEngin.Drag = Drag
# PropuOpti = MDOEngin.add_driver(Optimizer('optim',method='COBYLA'))
# PropuOpti.add_unknown('IntakeDiameter', lower_bound=0, upper_bound=3)
# PropuOpti.add_unknown('TCombustion', lower_bound=500, upper_bound=3500)
# PropuOpti.add_unknown('Teta', lower_bound=0, upper_bound=1)
# PropuOpti.add_unknown('Number', lower_bound=1, upper_bound=5)

# PropuOpti.add_unknown(['IntakeDiameter','TCombustion','Teta','Number'])

# PropuOpti.add_constraints([
#     'Number - round(Number)=0',
#     'TCombustion <= 3500',
#     'Thrust>=Drag',
#     "IntakeDiameter>0"  #Comme si il ne prenais pas en compte cette partie 

# ])

# PropuOpti.set_minimum("IntakeDiameter")
# MDOEngin.run_drivers()


print("ok")