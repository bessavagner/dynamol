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
    def __init__(self, N, T=1.0, compress=1.0, dt=1.0e-3, dim=3,
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
        self.time_step = dt
        self.pressure = 0.0
        self.T = T/self.units.temperature
        if mass is None:
            mass = np.ones(self.N)

        density = self.check_inputs(compress)
        self.V = N*self.units.mass/density
        if self.dim == 2:
            self.V /= self.units.space**2
        else:
            self.V /= self.units.volume
        self.size = np.ones(self.dim)*self.V**(1/self.dim)

        self.celllist = construct.CellList(self.N, self.size, L=cutoff)

        self.initialize(mass, config_file)

        self.interaction = forcelaws.LennardJones(cutoff=self.cutoff)
        self.integration = methods.VelocityVerlet(dt=dt)

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
        print(f"\t\t Densidade: {density} kg/m³ or kg/m²")
        print(f"\t\t Espaçamento inicial: {(self.V/N)**(1/dim):.3f} uL")

    def initialize(self, mass, config_file=None):
        if config_file is not None:
            pass
        else:
            R, self.N, self.dim = self.celllist.square_lattice()
            V = self.static_system(mass)
            super().__init__(self.N, self.dim)
            self.positions = R
            self.velocities = V

    def update_list(self, ):
        r = np.array([r for r in self.positions])
        self.celllist.make_list(r)

    def compute_interactions(self, ):
        self.check_reflections()
        self.update_list()
        cum_forces = np.zeros((self.N, self.dim))
        for loc in self.celllist.index_list():
            neighbor = self.celllist.neighbors(loc)
            if len(neighbor) <= 1:
                break
            R = np.array([p.r for p in self[neighbor]])
            V = np.array([p.v for p in self[neighbor]])
            A = np.array([p.a for p in self[neighbor]])
            relative_positions = construct.compute_differences(
                np.array(R)
            )
            interactions = np.array([
                self.interaction.force(r, neighbor) for r in relative_positions
            ]).reshape(len(neighbor), len(neighbor) - 1, self.dim)
            cum_forces[neighbor] += np.sum(interactions, axis=1)

        masses = np.array([p.m for p in self[:]])
        R = np.array([p.r for p in self[:]])
        V = np.array([p.v for p in self[:]])
        A = np.array([p.a for p in self[:]])

        def accels():
            return np.divide(cum_forces, masses[:, None])

        R, V, A = self.integration.single_step((R, V), A, accels)
        for p, r, v, a in zip(self[:], R, V, A):
            p.r, p.v, p.a = r, v, a

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
        self.compute_interactions()  # start accelerations
        self.store_variables(time=0, maxlines=n_files+1)
        self.save_positions(idx=0)

        for t in trange(n_intereations):
            self.compute_interactions()
            # self.check_reflections()
            if t % file_ratio == 0:
                idx = int((t + 1)/file_ratio)
                self.save_positions(idx=idx, zeros=zeros)
                self.store_variables(time=(t + 1)*self.integration.dt,
                                     idx=(idx+1))

    def check_reflections(self, ):
        self.pressure = 0.0
        for p in self[:]:
            for i, (u, v, l) in enumerate(zip(p.r, p.v, self.size)):
                if u <= 0.0:
                    p.v[i] = -p.v[i]
                    p.r[i] = 0.0
                    self.pressure += l*p.m*abs(p.v[i])/self.time_step
                elif u >= l:
                    p.v[i] = -p.v[i]
                    p.r[i] = l
                    self.pressure += l*p.m*abs(p.v[i])/self.time_step
        self.pressure /= 3*self.V

    def store_variables(self, time, idx=0, maxlines=10000):
        file = self.vars_folder + r'\variables.h5'
        U = self.potential_energy()
        K = self.kinetic_energy()
        data = {
            'Mechanical Energy': K + U,
            'Potential Energy': U,
            'Kinetic Energy': K,
            'Pressure': self.pressure*self.units.pressure/1.0e5,
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
        for loc in self.celllist.index_list():
            neighbor = self.celllist.neighbors(loc)
            R = np.array([p.r for p in self[neighbor]])
            if len(neighbor) <= 1:
                break
            relative_positions = construct.compute_differences(
                np.array(R)
            )
            U += np.sum(
                np.array(
                    [self.interaction.potential(
                        np.linalg.norm(r)
                    ) for r in relative_positions]
                )
            )
        return U

    def kinetic_energy(self, ):
        """Calcula energia cinética e temperatura usando
        o teorema da equipartição: K = 3kT/2, k = 1 (Boltzmann)

        Returns:
            némero: valor da energia cinética
        """
        K = 0
        for p in self.particles:
            K += p.m*p.v.dot(p.v)
        self.T = K/(self.dim*self.N)
        K *= 0.5
        return K

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
        T = V2/(self.dim*self.N)
        k = np.sqrt(self.T/T)  # reescaling temperature
        self.T = T
        V *= k
        return V

    def check_inputs(self, compress):

        nice_dt = 1.0e-2/np.sqrt(self.T*self.units.temperature)
        if self.time_step > nice_dt:
            self.time_step = nice_dt
            print(f"Espaçamento no tempo ajustado para {nice_dt}")

        # Cálculo e ajuste da densidade
        N = self.N
        n_mol = N*self.units.Avogadro
        V = data.vol_mol_constant**int(self.dim/3)  # m³/mol
        m = n_mol*self.units.mass  # kg.mol
        density = compress**self.dim*(m/V)  # kg/m³

        while((self.units.mass/density)**(1/self.dim)/self.units.space <= 1):
            compress /= 1.01
            density = compress**self.dim*m/V  # kg/m³

        return density
