import numpy as np

def phase_plot(f, ax, /, *, domain=None, classic=False, n_points=500):
    theta = -np.pi
    domain = [-1, 1, -1, 1] if domain is None else domain
    
    x = np.linspace(domain[0], domain[1], num=n_points)
    y = np.linspace(domain[2], domain[3], num=n_points)
    
    [xx, yy] = np.meshgrid(x, y)
    zz = xx + yy*1j
    
    if classic:
        phi = lambda t: t
    else:
        phi = lambda t:  t - .5*np.cos(1.5*t)**3*np.sin(1.5*t)
    
    colors_values = np.mod(phi( np.angle(f(zz)) - np.pi)+ np.pi - theta, 2*np.pi) + theta
    # TODO: Work out if the color shift is needed
    im = ax.imshow(colors_values, origin="lower", extent=domain, cmap="hsv", vmin=theta, vmax=theta+2*np.pi)
    return im