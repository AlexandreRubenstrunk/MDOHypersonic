import numpy as np

class Engine():

    def __init__(self):
        self.A_exit=float()

def Thrust(Engine_Studied:Engine,debit_m_inlet, Speed_inlet, P_inlet):
    """this function aime to calculte the thrust generate by the studied reactor"""

    F = debit_m_exit * Speed_exit - debit_m_inlet * Speed_inlet + (P_exit-P_inlet) * Engine_Studied.A_exit