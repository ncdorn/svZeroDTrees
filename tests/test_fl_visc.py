import math
import matplotlib.pyplot as plt

def fl_visc(diameter):
    '''
    calculate the viscosity within a vessel of diameter < 300 um based on empirical relationship describing 
    fahraeus-lindqvist effect

    :param diameter: vessel diameter in um
    '''
    H_d = 0.45 # hematocrit
    u_45 = 6 * math.exp(-0.085 * diameter) + 3.2 - 2.44 * math.exp(-0.06 * diameter ** 0.645)
    C = (0.8 + math.exp(-0.075 * diameter)) * (-1 + (1 + 10 ** -11 * diameter ** 12) ** -1) + (1 + 10 ** -11 * diameter ** 12) ** -1
    rel_viscosity = (1 + (u_45 - 1) * (((1 - H_d) ** C - 1) / ((1 - 0.45) ** C - 1)) * (diameter / (diameter - 1.1)) ** 2) * (diameter / (diameter - 1.1)) ** 2

    viscosity = .012 * rel_viscosity

    return rel_viscosity


def plot_fl_visc():
    '''
    plot the fahraeus-lindqvist effect
    '''
    diameters = [x * 10 for x in range(1, 1000)]
    viscosities = [fl_visc(d) for d in diameters]

    plt.plot(diameters, viscosities)
    plt.xlabel('diameter (um)')
    plt.xscale('log')
    plt.ylabel('viscosity (dynes/cm2)')
    plt.show()


if __name__ == '__main__':
    
    plot_fl_visc()