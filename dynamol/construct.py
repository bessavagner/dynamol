import numpy as np
from dynamol import data
import scipy.constants as sc
from functools import reduce


class Particle:
    def __init__(self, r=[], v=[], a=[], m=None, dim=3):
        """Clase base de uma partícula

        Args:
            r (3D/2D array): posição
            v (3D/2D array): velocidade
            a (3D/2D array): aceleração
            m (int, optional): massa. Padrão: 1.
        """
        if len(r) == 0:
            if dim == 2:
                r = data.zero2D
            else:
                r = data.zero3D
        if len(v) == 0:
            if dim == 2:
                v = data.zero2D
            else:
                v = data.zero3D
        if len(a) == 0:
            if dim == 2:
                a = data.zero2D
            else:
                a = data.zero3D

        if dim not in (2, 3):
            clsname = type(self).__name__
            print(f"{clsname}: Dimensão {dim} inválida. Usando 3.")
            dim = 3

        self.dim = dim
        self.r = np.array(r)
        self.v = np.array(v)
        self.a = np.array(a)
        if m is None:
            m = 1
        self.m = m

    def __repr__(self, ):
        text = "Partícula: \n"
        text += f"\tPosição: {self.r}\n"
        text += f"\tVelocidade: {self.v}\n"
        return text


class SetOfParticles:
    """Clase base para um conjunto de partículas
    """
    def __init__(self, N=27, dim=3):
        self.N = N
        self.dim = dim
        self.particles = [Particle(dim=dim) for n in range(N)]
        self.particles = np.array(self.particles, dtype=object)

    def __getitem__(self, n):
        return self.particles[n]

    def __setitem__(self, n, item: Particle):
        self.particles[n] = item

    @property
    def positions(self, ):
        for p in self.particles:
            yield p.r

    @property
    def velocities(self, ):
        for p in self.particles:
            yield p.v

    @property
    def accels(self, ):
        for p in self.particles:
            yield p.a

    @property
    def masses(self, ):
        for p in self.particles:
            yield p.m

    @positions.setter
    def positions(self, R):
        for p, r in zip(self.particles, R):
            p.r = np.array(r)

    @velocities.setter
    def velocities(self, V):
        for p, v in zip(self.particles, V):
            p.v = np.array(v)

    @accels.setter
    def accels(self, A):
        for p, a in zip(self.particles, A):
            p.a = np.array(a)

    @masses.setter
    def masses(self, M):
        for p, m in zip(self.particles, M):
            p.m = m


class Lattice:
    def __init__(self, N: int, size=(1, 1, 1)):
        """Constrói uma rede retangular de N sítio e
    dimensões size

        Args:
            N (int): número de sítios
            size (tuple, optional): dimensões. Defaults to (1, 1, 1).
        """
        self.N = N
        self.size = np.array(size)
        self.dim = len(size)

    def square_lattice(self,):
        """Rede quadrada

        Returns:
            np.array, tuple: rede, dimensions
        """
        volume = reduce(lambda a, b: a*b, self.size)
        self.lattice_parameter = (volume/self.N)**(1/self.dim)
        N = [int(np.round(s/self.lattice_parameter)) for s in self.size]
        N_new = reduce(lambda a, b: a*b, N)
        if N_new != self.N:
            text = "Número de pontos ajustado para corresponder"
            text += " a uma rede regular:"
            text += f" {self.N} -> {N_new}"
            print(text)
            self.N = N_new
        r = [np.linspace(0.0, s, n) for s, n in zip(self.size, N)]
        r = np.array(r)
        R = np.meshgrid(*r)
        self.r = np.vstack([u.ravel() for u in R]).T
        return self.r, self.N, r.shape[0]


class Cells:
    def __init__(self, shape=(1, 1, 1)):
        """Células para preenchimento de objetos

        Args:
            shape (tuple, optional): formato do bloco
                de células. Defaults to (1, 1, 1).
        """
        self.cells = np.empty(shape, dtype=list)
        self.shape = shape
        self.dim = len(shape)
        self.clean()
        indices = tuple(range(a) for a in shape)
        index_splitted = np.meshgrid(*indices)
        self.index_list = np.vstack([i.ravel() for i in index_splitted]).T

    def add(self, item, pos=(0, 0, 0)):
        """Adiciona o objeto 'item' à coordenada de bloco
        de célula 'pos'

        Args:
            item (object): item a ser adicionado
            pos (tuple, optional): indice. Defaults to (0, 0, 0).
        """
        if self.cells[pos] is None:
            self[pos] = [item]
        else:
            if item not in self.cells[pos]:
                self.cells[pos].append(item)

    def clean(self, ):
        for i in range(self.cells.shape[0]):
            for j in range(self.cells.shape[1]):
                if self.dim == 3:
                    for k in range(self.cells.shape[2]):
                        self.cells[i, j, k] = []
                else:
                    self.cells[i, j] = []

    def __getitem__(self, loc: tuple):
        return self.cells[loc]

    def __setitem__(self, loc: tuple, item):
        self.cells[loc] = item

    def __repr__(self, ):
        text = ""
        for i, a in enumerate(self.cells):
            for j, b in enumerate(a):
                if self.dim == 2:
                    text += f"({i}, {j}) -> {b}\n"
                else:
                    for k, c in enumerate(b):
                        text += f"({i}, {j}, {k}) -> {c}\n"
        return text


class CellList(Lattice):
    def __init__(self, N: int, size=(20, 20, 20), L=3.0):
        self.nc = np.array([int(np.ceil(s/L)) for s in size])
        self.Nc = reduce(lambda a, b: a*b, self.nc)
        self.L = L
        super().__init__(N, size)
        self.cells = Cells(self.nc)

    def make_list(self, points: np.ndarray):
        self.cells.clean()
        try:
            mask = np.floor_divide(points, self.L).astype(int)
            if np.array(points).max()/self.L >= self.nc.max():
                mask[mask == mask.max()] = self.nc.max() - 1
            for i, m in enumerate(mask):
                self.cells.add(i, tuple(m))
        except IndexError:
            print(points, len(points))
            import sys
            sys.exit()
        return self

    def index_list(self, ):
        return self.cells.index_list

    def neighbors(self, loc: tuple):
        idx = np.empty(2*len(self.cells.shape), dtype=int)
        for n, i in enumerate(loc):
            if i <= 0:
                idx[2*n] = 0
            else:
                idx[2*n] = i - 1
            if i >= self.cells.shape[n]:
                idx[2*n + 1] = self.cells.shape[n]
            else:
                idx[2*n + 1] = i + 2
        idx_arr = np.resize(idx, (len(loc), 2))
        slices = tuple(slice(a, b) for a, b in idx_arr)
        return np.sum(self.cells[slices].flatten())


class SystemOfUnits:
    def __init__(self, mass=data.atomic_mass['argon'],
                 energy=data.dispersion_energy['argon'],
                 space=data.spacing['argon']):
        """Sistema de unidades

        Args:
            mass (número, optional): Defaults to atomic_mass['argon'].
            energy (número, optional): Defaults to s['argon'].
            space (número, optional): Defaults to spacing['argon'].
        """
        self.mass = mass
        self.energy = energy
        self.space = space
        self.Avogadro = sc.Avogadro
        self.Boltzmann = sc.Boltzmann

    @property  # decorador
    def volume(self, ):
        return self.space**3

    @property
    def density(self, ):
        return self.mass/self.volume

    @property
    def time(self, ):
        return np.sqrt(self.mass/self.energy)*self.space

    @property
    def temperature(self, ):
        return self.energy/self.Boltzmann

    @property
    def pressure(self, ):
        return self.energy/self.volume

    def set_units(self, atom='argon'):
        """Modifica o sistema de unidade

        Args:
            atom (str, optional): chave de base.atomic_mass, e.g.
            Defaults to 'argon'.

        Returns:
            SystemOfUnits: novo sistema de unidades
        """
        self.mass = data.atomic_mass[atom]
        self.energy = data.dispersion_energy[atom]
        self.space = data.spacing[atom]
        return self

    def __repr__(self):
        unidades = ["Sistema de unidades:\n",
                    f"\tmassa: {self.mass} kg.\n"
                    f"\tenergia: {self.energy} J.\n",
                    f"\tespaço: {self.space} m.\n",
                    f"\tvolume: {self.volume} m³.\n",
                    f"\tdensidade: {self.density} kg/m³.\n",
                    f"\ttempo: {self.time} s.\n",
                    f"\ttemperatura: {self.temperature} K.\n",
                    f"\tpressão: {self.pressure} Pa.\n"
                    ]
        return reduce(lambda a, b: a + b, unidades)


def pair_difference_matrices(N):
    A = np.zeros((N, N - 1), dtype=int)
    B = (N + 1)*np.ones((N, N - 1), dtype=int)
    F = np.tril(np.ones((N - 1, N), dtype=int)).T
    f = -np.tril(np.ones((N, N - 1), dtype=int), k=-1)
    C = f + F
    for i in range(N):
        A[i:, i:] = A[i:, i:] + np.ones(N - i - 1, dtype=int)
        B[i:, i:] = B[i:, i:] - np.ones(N - i - 1, dtype=int)
    B = np.rot90(np.rot90(B))
    A -= 1
    B -= 1
    return A.flatten(), B.flatten(), C.flatten()


def compute_differences(r: np.ndarray):
    """Calcula as diferenças entre os pares de
    vetores em r

    Args:
        r (np.ndarray): vetores posição

    Returns:
        np.ndarray: matriz posições relativas.
            cada linha i representa a posição
            relativa do ponto i a todos os outros.
    """
    N = len(r)
    dim = r[0].shape[0]
    if N == 1:
        return r
    else:
        M = pair_difference_matrices(len(r))
        d = np.subtract(r[M[0]], r[M[1]])
        d = np.multiply(d, M[2][:, None])
        d = d.flatten().reshape(N*(N - 1), dim)
        return d
