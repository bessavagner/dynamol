import numpy as np
from dynamol import construct
array = np.ndarray


class Integrator:
    def __init__(self, dt=1.0e-4):
        self.__dt = dt
        self.__halfdt = 0.5*dt
        self.__halfdt2 = self.__halfdt*dt

    @property
    def dt(self, ):
        return self.__dt

    @property
    def halfdt(self, ):
        return self.__halfdt

    @property
    def halfdt2(self, ):
        return self.__halfdt2

    @dt.setter
    def dt(self, dt):
        self.__dt = dt
        self.__halfdt = 0.5*dt
        self.__halfdt2 = self.__halfdt*dt


class VelocityVerlet(Integrator):
    def __init__(self, dt=1.0e-4):
        super().__init__(dt)

    def __update_position(self, r, v, a):
        return r + v*self.dt + a*self.halfdt2

    def __update_velocity(self, v, a, a_old):
        return v + (a + a_old)*self.halfdt

    def single_step(self, r, v, a_old,
                    f: callable, *args, **kwargs):
        r = self.__update_position(r, v, a_old)
        a = f(r, *args, **kwargs)
        v = self.__update_velocity(v, a, a_old)
        return r, v, a
