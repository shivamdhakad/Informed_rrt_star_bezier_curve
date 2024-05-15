
class Env:

    def __init__(self):
        r = 10
        self.x_range = (0, 10 * r)
        self.y_range = (0, 10 * r)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()

    @staticmethod
    def obs_boundary():
        r = 10
        obs_boundary = [
            [0, 0, 0.1*r, 10*r],
            [0, 10*r, 10*r, 0.1*r],
            [0.1*r, 0, 10*r, 0.1*r],
            [10*r, 0.1*r, 0.1*r, 10*r]
        ]
        return obs_boundary

    @staticmethod
    def obs_circle():
        r = 10
        obs_cir = [
            [2.7*r, 4*r, 0.22*r],
            [1*r, 0.3*r, 0.22*r],
            [1.6*r, 0.6*r, 0.22*r],
            [1.2*r, 4.2*r, 0.3*r],
            [2.1*r, 1.8*r, 0.3*r],
            [0.5*r, 5*r, 0.22*r],
            [3.1*r, 2.0*r, 0.3*r],
            [3.3*r, 1.0*r, 0.22*r],
            [4*r, 2*r, 0.22*r],
            [4.1*r, 7.5*r, 0.22*r],
            [4.6*r, 6.7*r, 0.22*r],
            [3.7*r, 5.3*r, 0.35*r],
            [5.2*r, 1.2*r, 0.25*r],
            [6.8*r, 1.6*r, 0.23*r],
            [5.0*r, 4.6*r, 0.4*r],
            [5.5*r, 7.2*r, 0.22*r],
            [5.7*r, 8.9*r, 0.25*r],
            [5.3*r, 8.4*r, 0.22*r],
            [6.0*r, 6.4*r, 0.35*r],
            [6.6*r, 7.8*r, 0.35*r],
            [7.1*r, 4.8*r, 0.22*r],
            [8.0*r, 3.4*r, 0.35*r],
            [8.1*r, 7.5*r, 0.22*r],
            
            [8.2*r, 6.1*r, 0.22*r],
            [3.2*r, 2.7*r, 0.35*r],

        ]

        return obs_cir