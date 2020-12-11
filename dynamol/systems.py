import h5py
import numpy as np
from tqdm import tqdm  # barra de progresso
from dynamol import construct
from dynamol import forcelaws
from dynamol import methods
from dynamol import data
from dynamol import files


def trange(N):
    return tqdm(range(N))


class IdealGas(construct.SetOfParticles):
    def __init__(self, N, T=1.0, compress=1.0, dt=2.0e-15, dim=3,
                 atom='argon', cutoff=3.0, mass=None, config_file=None,
                 folder='outputs'):
        files.mkdir(folder)
        self.folder = folder
        self.position_folder = files.mkdir(folder + r'\positions')
        self.vars_folder = files.mkdir(folder + r'\variables')
        self.units = construct.SystemOfUnits().set_units(atom)
        self.dim = dim
        self.N = N
        self.cutoff = cutoff
        self.time_step = dt/self.units.time
        self.pressure = 0.0
        self.T = T/self.units.temperature
        self.T_bath = T
        self.tau = 1.0e5*self.time_step
        if mass is None:
            mass = np.ones(self.N)

        density = self.check_inputs(compress)
        self.V = N*self.units.mass/density
        self.density = density/self.units.density
        if self.dim == 2:
            self.V /= self.units.space**2
        else:
            self.V /= self.units.volume
        self.size = np.ones(self.dim)*self.V**(1/self.dim)

        self.cellist = construct.CellList(self.N, self.size, L=cutoff)

        self.initialize(mass, config_file)

        self.interaction = forcelaws.LennardJones(cutoff=self.cutoff)
        self.integration = methods.VelocityVerlet(dt=self.time_step)

        print("\tIdeal gas")
        print(f"\t\t Número de partículas: {self.N}")
        print(f"\t\t Volume: {self.V*self.units.volume:.2e} m³.")
        print(f"\t\t Temperatura: {self.T*self.units.temperature} K")
        dim1 = f"{self.size[0]:.2e} x {self.size[1]:.2e} "
        if dim == 3:
            dim1 += f"x {self.size[2]:.2e}"
        uL = self.units.space
        dim2 = f"{self.size[0]*uL:.2e} x {self.size[1]*uL:.2e}"
        if dim == 3:
            dim2 += f" x {self.size[2]*uL:.2e}"
        print(f"\t\t Dimensões:\n\t\t  {dim1} uL³/uL², ou\n\t\t {dim2} m³/m²")
        print(f"\t\t Densidade: {density} kg/m³ ou kg/m² ou:\t\t")
        print(f"\t\t\t\t{self.N/self.V:.2f} partículas por uV")
        print(f"\t\t Espaçamento inicial: {(self.V/N)**(1/dim):.3f} uL")

    def initialize(self, mass, config_file=None):
        if config_file is not None:
            pass
        else:
            R, self.N, self.dim = self.cellist.square_lattice()
            V = self.static_system(mass)
            super().__init__(self.N, self.dim)
            self.positions = R
            self.velocities = V

    def update_list(self, r=None):
        if r is None:
            r = np.array([r for r in self.positions])
        else:
            self.positions = r
        self.cellist.make_list(r)

    def compute_accels(self, r):
        self.update_list(r)
        cum_forces = np.zeros((self.N, self.dim))
        for loc in self.cellist.index_list():
            for i in self.cellist.cells[tuple(loc)]:
                for j in self.cellist.neighbors(tuple(loc)):
                    if i != j:
                        rij = self[i].r - self[j].r
                        cum_forces[i] += self.interaction.force(rij)

        masses = np.array([p.m for p in self[:]])
        return np.divide(cum_forces, masses[:, None])

    def compute_interactions(self, time):

        R = np.array([p.r for p in self[:]])
        V = np.array([p.v for p in self[:]])
        A = np.array([p.a for p in self[:]])

        accels = self.compute_accels

        R, V, A = self.integration.single_step(R, V, A, accels)
        for p, r, v, a in zip(self.particles, R, V, A):
            p.r, p.v, p.a = r, v, a
        self.check_reflections(time)

    def execute_simulation(self, n_intereations, start=0,
                           n_files=1000, zeros=4):
        if n_files > n_intereations:
            n_files = n_intereations
        file_ratio = int(np.ceil(n_intereations/n_files))
        while(len(str(n_files)) > zeros):
            zeros += 1
        text = "Iniciando simulação...\n"
        text += f"\tArmanezando dados a cada {file_ratio} passos."
        print(text)
        self.compute_interactions(self.time_step)  # start accelerations
        self.store_variables(time=0, maxlines=n_files+1)
        self.save_positions(idx=0)
        time = 0.0
        for t in trange(n_intereations):
            time += self.integration.dt
            self.compute_interactions(time)
            if t % file_ratio == 0:
                idx = int((t + 1)/file_ratio)
                self.save_positions(idx=idx, zeros=zeros)
                self.store_variables(time=time,
                                     idx=(idx+1))

    def check_reflections(self, time):
        pressure = 0.0
        for p in self[:]:
            for i, (u, v, l) in enumerate(zip(p.r, p.v, self.size)):
                if u < 0.0:
                    p.v[i] = -p.v[i]
                    p.r[i] = 0.0
                    pressure += l*p.m*abs(p.v[i])/self.time_step
                elif u > l:
                    p.v[i] = -p.v[i]
                    p.r[i] = l
                    pressure += l*p.m*abs(p.v[i])/self.time_step
        pressure /= 3*self.V
        self.pressure += pressure

    def store_variables(self, time, idx=0, maxlines=10000):
        file = self.vars_folder + r'\variables.h5'
        U = self.potential_energy()*self.units.kJmol/self.N
        K = self.kinetic_energy()*self.units.kJmol/self.N
        data = {
            'Mechanical Energy': K + U,
            'Potential Energy': U,
            'Kinetic Energy': K,
            'Average Pressure': self.pressure*self.units.pressure/1.0e5,
            'Temperature': self.T*self.units.temperature,
        }
        if idx == 0:
            maxshape = (maxlines, len(data))
            with h5py.File(file, 'w') as f:
                for key in data:
                    if key not in f.keys():
                        f.create_dataset(key, shape=(1, 2),
                                         maxshape=maxshape, dtype='float64')
                    f[key][:] = np.array((time, data[key]))
        else:
            with h5py.File(file, 'a') as f:
                for key in data:
                    f[key].resize(idx + 1, axis=0)
                    f[key][idx:] = np.array((time, data[key]))

    def save_positions(self, idx: int, zeros=5):
        fname = self.position_folder + \
            fr"\positions_{str(idx).zfill(zeros)}.h5"
        with h5py.File(fname, 'w') as f:
            r = np.array([u for u in self.positions])
            f.create_dataset('positions', shape=r.shape)
            f['positions'][:] = r

    def potential_energy(self, ):
        U = 0
        self.update_list()
        for loc in self.cellist.index_list():
            for i in self.cellist.cells[tuple(loc)]:
                for j in self.cellist.neighbors(tuple(loc)):
                    if i != j:
                        rij = self[j].r - self[i].r
                        U += self.interaction.potential(
                            np.linalg.norm(rij)
                        )
        return U

    def kinetic_energy(self, ):
        """Calcula energia cinética e temperatura usando
        o teorema da equipartição: K = self.dim*k*T/2, k = 1 (Boltzmann)

        Returns:
            némero: valor da energia cinética
        """
        K = 0
        for p in self[:]:
            K += p.m*p.v.dot(p.v)
        self.T = K/(self.dim*(self.N - 1))
        K *= 0.5
        return K

    def thermic_bath(self, ):
        self.kinetic_energy()
        T_factor = np.sqrt(1 + (self.time_step/self.tau)
                           * (self.T_bath/self.T - 1.0))
        self.velocities = T_factor*np.array([v for v in self.velocities])
        self.kinetic_energy()

    def static_system(self, mass=None):
        """Configura as velocidades

        Args:
            mass (sequência/número , optional):  Defaults to None.

        Returns:
            [type]: [description]
        """
        V = np.random.normal(scale=np.sqrt(self.T),
                             size=(self.N, self.dim))
        Vcm, M, V2 = 0, 0, 0
        if mass is None:
            mass = 1.0
        for m, v in zip(mass, V):
            Vcm += m*v
            M += m
        V -= Vcm/M
        for m, v in zip(mass, V):
            V2 += m*v.dot(v)
        T = V2/(self.dim*(self.N - 1))
        k = np.sqrt(self.T/T)  # reescaling temperature
        self.T = T
        V *= k
        return V

    def spacing(self, density):
        return (self.units.mass/density)**(1/self.dim)/self.units.space

    def check_inputs(self, compress):

        # Cálculo e ajuste da densidade
        N = self.N
        n_mol = N*self.units.Avogadro
        V = data.vol_mol_constant**int(self.dim/3)  # m³/mol
        m = n_mol*self.units.mass  # kg.mol
        density = compress**self.dim*(m/V)  # kg/m³

        while(self.spacing(density) <= (2)**(1/6)):
            compress /= 1.01
            density = compress**self.dim*m/V  # kg/m³

        return density
