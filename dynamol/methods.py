class Integrator:
    def __init__(self, dt=1.0e-4):
        self.dt = dt
        self.halfdt = 0.5*dt
        self.halfdt2 = self.halfdt*dt


class VelocityVerlet(Integrator):
    def __init__(self, dt=1.0e-4):
        super().__init__()

    def __update_position(self, state_vector, a):
        r, v = state_vector
        return r + v*self.dt + a*self.halfdt2

    def __update_velocity(self, v, a, a_old):
        return v + (a + a_old)*self.halfdt

    def single_step(self, state_vector, a,
                    f: callable, *args, **kwargs):
        r = self.__update_position(state_vector, a)
        a_new = f(*args, **kwargs)
        v = self.__update_velocity(state_vector[1], a_new, a)
        return r, v, a_new
