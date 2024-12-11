import numpy as np


class Quintic:
    """
    Represents a quintic (5th-degree) polynomial trajectory with precise start and end conditions.

    Args:
        s (float): Starting position
        ss (float): Starting velocity
        sss (float): Starting acceleration
        e (float): Ending position
        ee (float): Ending velocity
        eee (float): Ending acceleration
        t (float): current time

    Methods:
        calc_point(t): Compute position at time t
        calc_first_derivative(t): Compute velocity at time t
        calc_second_derivative(t): Compute acceleration at time t
        calc_third_derivative(t): Compute jerk at time t
    """

    def __init__(self, s, ss, sss, e, ee, eee, t):
        self.coeffs = self._compute_coefficients(s, ss, sss, e, ee, eee, t)
        self.first_deriv_coeffs = np.polyder(self.coeffs, 1)
        self.second_deriv_coeffs = np.polyder(self.coeffs, 2)
        self.third_deriv_coeffs = np.polyder(self.coeffs, 3)

    def _compute_coefficients(self, s, ss, sss, e, ee, eee, t):
        a0, a1, a2 = s, ss, sss / 2.0
        M = np.array([[t**3, t**4, t**5], [3 * t**2, 4 * t**3, 5 * t**4], [6 * t, 12 * t**2, 20 * t**3]])
        ζ = np.array([e - a0 - a1 * t - a2 * t**2, ee - a1 - 2 * a2 * t, eee - 2 * a2])
        a3, a4, a5 = np.linalg.solve(M, ζ)
        return np.array([a0, a1, a2, a3, a4, a5])

    def calc_point(self, t):
        return np.polynomial.polynomial.polyval(t, self.coeffs)

    def calc_first_derivative(self, t):
        return np.polynomial.polynomial.polyval(t, self.first_deriv_coeffs)

    def calc_second_derivative(self, t):
        return np.polynomial.polynomial.polyval(t, self.second_deriv_coeffs)

    def calc_third_derivative(self, t):
        return np.polynomial.polynomial.polyval(t, self.third_deriv_coeffs)


class Quartic:
    """
    Represents a quartic (4th-degree) polynomial trajectory with precise start and end conditions.

    Args:
        s (float): Starting position
        ss (float): Starting velocity
        sss (float): Starting acceleration
        ee (float): Ending velocity
        eee (float): Ending acceleration
        t (float): current time

    Methods:
        calc_point(t): Compute position at time t
        calc_first_derivative(t): Compute velocity at time t
        calc_second_derivative(t): Compute acceleration at time t
        calc_third_derivative(t): Compute jerk at time t
    """

    def __init__(self, s, ss, sss, ee, eee, t):
        self.coeffs = self._compute_coefficients(s, ss, sss, ee, eee, t)
        self.first_deriv_coeffs = np.polyder(self.coeffs, 1)
        self.second_deriv_coeffs = np.polyder(self.coeffs, 2)
        self.third_deriv_coeffs = np.polyder(self.coeffs, 3)

    def _compute_coefficients(self, s, ss, sss, ee, eee, t):
        a0, a1, a2 = s, ss, sss / 2.0
        M = np.array([[3 * t**2, 4 * t**3], [6 * t, 12 * t**2]])
        ζ = np.array([ee - a1 - 2 * a2 * t, eee - 2 * a2])
        a3, a4 = np.linalg.solve(M, ζ)
        return np.array([a0, a1, a2, a3, a4])

    def calc_point(self, t):
        return np.polynomial.polynomial.polyval(t, self.coeffs)

    def calc_first_derivative(self, t):
        return np.polynomial.polynomial.polyval(t, self.first_deriv_coeffs)

    def calc_second_derivative(self, t):
        return np.polynomial.polynomial.polyval(t, self.second_deriv_coeffs)

    def calc_third_derivative(self, t):
        return np.polynomial.polynomial.polyval(t, self.third_deriv_coeffs)
