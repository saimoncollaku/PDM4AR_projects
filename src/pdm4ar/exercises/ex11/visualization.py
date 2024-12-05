from matplotlib import pyplot as plt
import numpy as np


class Visualizer:

    def __init__(self, bounds, r_s, planets, satellites, params):
        self.bounds = bounds
        self.r_s = r_s
        self.planets = planets
        self.satellites = satellites
        self.params = params

        self.global_fig, self.global_ax = plt.subplots(figsize=(36, 25), dpi=120)
        self.global_ax.set_xlim([self.bounds[0], self.bounds[2]])
        self.global_ax.set_ylim([self.bounds[1], self.bounds[3]])
        for name, planet in self.planets.items():
            planet = plt.Circle(planet.center, planet.radius, color="green")
            self.global_ax.add_patch(planet)

    def vis_iter(self, iteration, X, p):
        fig, ax = plt.subplots(figsize=(36, 25), dpi=120)
        ax.set_xlim([self.bounds[0], self.bounds[2]])
        ax.set_ylim([self.bounds[1], self.bounds[3]])
        for name, planet in self.planets.items():
            planet = plt.Circle(planet.center, planet.radius, color="green")
            ax.add_patch(planet)
            planet = plt.Circle(planet.center, planet.radius + self.r_s, color="red", alpha=0.2)
            ax.add_patch(planet)

        for name, satellite in self.satellites.items():
            planet_name = name.split("/")[0]
            for k in range(self.params.K):
                t = k / self.params.K
                θ = satellite.omega * p[0] * t + satellite.tau
                Δθ = np.array([np.cos(θ), np.sin(θ)])
                # print(θ, Δθ)
                satellite_center = self.planets[planet_name].center + satellite.orbit_r * Δθ
                alpha = 1 if k == 0 else 0.2
                satellite_k = plt.Circle(
                    satellite_center, satellite.radius, color=plt.cm.viridis(k / self.params.K), alpha=alpha
                )
                ax.add_patch(satellite_k)
                satellite_k_glob = plt.Circle(
                    satellite_center,
                    satellite.radius,
                    color=plt.cm.viridis(k / self.params.K),
                    alpha=iteration / self.params.max_iterations,
                )
                self.global_ax.add_patch(satellite_k_glob)

        # for k in range(self.params.K):
        ax.plot(
            X[0, :],
            X[1, :],
        )
        ax.scatter(
            X[0, :],
            X[1, :],
            color=plt.cm.viridis(np.linspace(0, 1, self.params.K)),
            s=256,
        )

        self.global_ax.plot(
            X[0, :],
            X[1, :],
            linewidth=4,
            alpha=iteration / self.params.max_iterations,
        )

        for k in range(self.params.K):
            block = plt.Circle(X[0:2, k], self.r_s, alpha=0.1, color="grey")
            ax.add_patch(block)

        fig.savefig("../../out/11/vis_" + str(iteration) + ".png", bbox_inches="tight")
        plt.close(fig)
        self.global_fig.savefig("../../out/11/vis_glob.png", bbox_inches="tight")

    def vis_k(self, iteration, X, p):
        for k in range(self.params.K):
            fig, ax = plt.subplots(figsize=(36, 25), dpi=120)
            ax.set_xlim([self.bounds[0], self.bounds[2]])
            ax.set_ylim([self.bounds[1], self.bounds[3]])
            for name, planet in self.planets.items():
                planet = plt.Circle(planet.center, planet.radius, color="green")
                ax.add_patch(planet)
                planet = plt.Circle(planet.center, planet.radius + self.r_s, color="red", alpha=0.2)
                ax.add_patch(planet)
            for name, satellite in self.satellites.items():
                planet_name = name.split("/")[0]
                t = k / self.params.K
                θ = satellite.omega * p[0] * t + satellite.tau
                Δθ = np.array([np.cos(θ), np.sin(θ)])
                satellite_center = self.planets[planet_name].center + satellite.orbit_r * Δθ
                satellite_k = plt.Circle(satellite_center, satellite.radius, color="green", alpha=1)
                ax.add_patch(satellite_k)
                satellite_k = plt.Circle(satellite_center, satellite.radius + self.r_s, color="red", alpha=0.2)
                ax.add_patch(satellite_k)
            if k < self.params.K - 1:
                ax.plot(
                    [
                        X[0, k],
                        X[0, k + 1],
                    ],
                    [
                        X[1, k],
                        X[1, k + 1],
                    ],
                    marker="x",
                    linewidth=8,
                    markersize=16,
                )

                block = plt.Circle(X[0:2, k], self.r_s, alpha=0.2, color="blue")
                ax.add_patch(block)
                block = plt.Circle(
                    X[0:2, k + 1],
                    self.r_s,
                    alpha=0.2,
                    color="blue",
                )
                ax.add_patch(block)

                fig.savefig("../../out/11/vis_k/vis_" + str(iteration) + "_" + str(k) + ".png", bbox_inches="tight")
                plt.close(fig)
