import numpy as np


class LennardJones:
    def __init__(self, cutoff=None):
        """Implementa a interação de Lennard-Jonnes adimensional

        Args:
            sigma (número, optional): . Defaults to 1..
            epsilon (número, optional): . Defaults to 1..
            cutoff (número, optional): raio de corte. Defaults to None.
        """
        if cutoff is None:
            self.cutoff = 3.0
        else:
            self.cutoff = cutoff
        self.F_cutoff = self.__force(self.cutoff)
        self.V_cutoff = self.__potential(self.cutoff)

    def force(self, rij, idx=0):
        """Força de Lennard-Jonnes deslocada

        Args:
            ri (np.array): posição
            rj (np.array): posição

        Returns:
            np.array: força
        """
        r = np.linalg.norm(rij)
        if r == 0:
            print(idx)
            import sys
            sys.exit()
        if r < self.cutoff:
            rm1 = 1.0/r
            rm6 = rm1**6
            rm7 = rm1*rm6
            return (48.0*rm7*(rm6 - 0.5) - self.F_cutoff)*rij/r
        else:
            return np.zeros(rij.shape[0])

    def potential(self, r, idx=0):
        """Força de Lennard-Jonnes deslocada

        Args:
            ri (np.array): posição
            rj (np.array): posição

        Returns:
            np.número: potencial
        """
        if r == 0:
            print(idx)
            import sys
            sys.exit()
        if r < self.cutoff:
            rm1 = 1.0/r
            rm6 = rm1**6
            return 4.0*rm6*(rm6 - 1.0) - self.F_cutoff*(r - self.cutoff)\
                - self.V_cutoff
        else:
            return 0

    def __force(self, r):
        """Força de Lennard-Jonnes

        Args:
            ri (np.array): posição
            rj (np.array): posição

        Returns:
            np.array: força
        """
        rm1 = 1.0/r
        rm6 = rm1**6
        rm7 = rm1*rm6
        return 48.0*rm7*(rm6 - 0.5)

    def __potential(self, r):
        """Força de Lennard-Jonnes

        Args:
            ri (np.array): posição
            rj (np.array): posição

        Returns:
            np.array: força
        """
        rm1 = 1.0/r
        rm6 = rm1**6
        return 4.0*rm6*(rm6 - 1.0)
