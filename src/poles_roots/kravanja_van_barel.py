from poles_roots.integration import argument_principle_from_points


def kravanja_van_barel(f, f_prime, points, N=None):
    if N is None:
        N = int(argument_principle_from_points(f, f_prime, points, moment=0)[0].real)


def kravanja_van_barel_FOP(f, f_prime, points, N=None):
    if N is None:
        N = int(argument_principle_from_points(f, f_prime, points, moment=0)[0].real)
