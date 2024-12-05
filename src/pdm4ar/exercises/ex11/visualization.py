from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os


class Visualizer:

    def __init__(self, bounds, sp, planets, satellites, params):
        self.bounds = bounds
        self.sg = sp
        self.planets = planets
        self.satellites = satellites
        self.params = params

        self.global_fig, self.global_ax = plt.subplots(figsize=(36, 25), dpi=120)
        self.global_ax.set_xlim([self.bounds[0], self.bounds[2]])
        self.global_ax.set_ylim([self.bounds[1], self.bounds[3]])
        self.global_ax.add_patch(
            Rectangle(
                (self.bounds[0] + self.sg.l / 2, self.bounds[1] + self.sg.l / 2),
                self.bounds[2] - self.bounds[0] - self.sg.l,
                self.bounds[3] - self.bounds[1] - self.sg.l,
                fill=False,
            )
        )
        self.global_ax.add_patch(
            Rectangle(
                (self.bounds[0] + self.sg.width / 2, self.bounds[1] + self.sg.width / 2),
                self.bounds[2] - self.bounds[0] - self.sg.width,
                self.bounds[3] - self.bounds[1] - self.sg.width,
                fill=False,
            )
        )
        for name, planet in self.planets.items():
            planet = plt.Circle(planet.center, planet.radius, color="green")
            self.global_ax.add_patch(planet)
        self.global_dock = False

    def vis_iter(self, iteration, X, U, p, kappa_planets, kappa_sats, dock_points):
        fig, ax = plt.subplots(figsize=(36, 25), dpi=120)
        ax.set_xlim([self.bounds[0], self.bounds[2]])
        ax.set_ylim([self.bounds[1], self.bounds[3]])
        ax.add_patch(
            Rectangle(
                (self.bounds[0] + self.sg.l / 2, self.bounds[1] + self.sg.l / 2),
                self.bounds[2] - self.bounds[0] - self.sg.l,
                self.bounds[3] - self.bounds[1] - self.sg.l,
                fill=False,
            )
        )
        min_kappa = self.sg.l * np.ones(self.params.K)
        for name, planet in self.planets.items():
            planet = plt.Circle(planet.center, planet.radius, color="green")
            ax.add_patch(planet)
            min_kappa = np.minimum(min_kappa, kappa_planets[name])
            # planet = plt.Circle(planet.center, planet.radius + self.sg.l, colfor="red", alpha=0.2)
            # ax.add_patch(planet)

        if dock_points is not None:
            A, B, C, A1, A2, pho = dock_points
            ax.scatter(A[0], A[1], s=1024, marker="x")
            ax.scatter(X[0, -1], X[1, -1], s=1024, c="r")
            ax.scatter(B[0], B[1], s=1024, marker="*")
            ax.scatter(C[0], C[1], s=1024, marker="*")
            ax.scatter(A1[0], A1[1], s=1024, marker="^")
            ax.scatter(A2[0], A2[1], s=1024, marker="^")
            ax.plot([A1[0], A2[0]], [A1[1], A2[1]], linewidth=16)
            if not self.global_dock:
                self.global_ax.plot([A1[0], A2[0]], [A1[1], A2[1]], linewidth=16)
                self.global_dock = True

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
                satellite_coll = plt.Circle(
                    satellite_center,
                    satellite.radius + kappa_sats[name][k],
                    color=plt.cm.viridis(k / self.params.K),
                    alpha=alpha,
                    fill=False,
                )
                # if k == 20:
                #     print("vis", name, satellite.radius, satellite.radius + kappa_sats[name][k], kappa_sats[name][k])
                ax.add_patch(satellite_coll)
                satellite_k_glob = plt.Circle(
                    satellite_center,
                    satellite.radius,
                    color=plt.cm.viridis(k / self.params.K),
                    alpha=min(2 * iteration / self.params.max_iterations, 1),
                )
                self.global_ax.add_patch(satellite_k_glob)

        # for k in range(self.params.K):
        ax.plot(
            X[0, :],
            X[1, :],
        )
        for k in range(self.params.K):
            ax.arrow(
                X[0, k] - self.sg.l_r * np.cos(X[2, k]),
                X[1, k] - self.sg.l_r * np.sin(X[2, k]),
                self.sg.l * np.cos(X[2, k]),
                self.sg.l * np.sin(X[2, k]),
                width=0.05,
                length_includes_head=True,
            )
            ax.arrow(
                X[0, k] - self.sg.l_r * np.cos(X[2, k]),
                X[1, k] - self.sg.l_r * np.sin(X[2, k]),
                U[0, k] * np.cos(U[1, k] + X[2, k]) / 2,
                U[0, k] * np.sin(U[1, k] + X[2, k]) / 2,
                width=0.03,
                color="r",
                length_includes_head=True,
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
            alpha=min(2 * iteration / self.params.max_iterations, 1),
        )
        for k in range(self.params.K):
            self.global_ax.arrow(
                X[0, k] - self.sg.l_r * np.cos(X[2, k]),
                X[1, k] - self.sg.l_r * np.sin(X[2, k]),
                self.sg.l * np.cos(X[2, k]),
                self.sg.l * np.sin(X[2, k]),
                width=0.05,
                length_includes_head=True,
                alpha=min(2 * iteration / self.params.max_iterations, 1),
            )
            self.global_ax.arrow(
                X[0, k] - self.sg.l_r * np.cos(X[2, k]),
                X[1, k] - self.sg.l_r * np.sin(X[2, k]),
                U[0, k] * np.cos(U[1, k] + X[2, k]) / 2,
                U[0, k] * np.sin(U[1, k] + X[2, k]) / 2,
                width=0.03,
                color="r",
                length_includes_head=True,
                alpha=min(2 * iteration / self.params.max_iterations, 1),
            )

        for k in range(self.params.K):
            block = plt.Circle(X[0:2, k], min_kappa[k], alpha=0.1, color="grey")
            ax.add_patch(block)

        savedir = "../../out/11/" + str(len(self.satellites)) + "_" + str(round(self.planets["Namek"].center[1], 2))
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        fig.savefig(
            savedir + "/vis_" + str(iteration) + ".png",
            bbox_inches="tight",
        )
        plt.close(fig)
        self.global_fig.savefig(
            savedir + "/vis_glob.png",
            bbox_inches="tight",
        )

    def vis_k(self, iteration, X, U, p, kappa_planets, kappa_sats):
        for k in range(self.params.K):
            fig, ax = plt.subplots(figsize=(36, 25), dpi=120)
            ax.set_xlim([self.bounds[0], self.bounds[2]])
            ax.set_ylim([self.bounds[1], self.bounds[3]])
            min_kappa = self.sg.l * np.ones(self.params.K)
            for name, planet in self.planets.items():
                planet = plt.Circle(planet.center, planet.radius, color="green")
                ax.add_patch(planet)
                min_kappa = np.minimum(min_kappa, kappa_planets[name])
            for name, satellite in self.satellites.items():
                planet_name = name.split("/")[0]
                t = k / self.params.K
                θ = satellite.omega * p[0] * t + satellite.tau
                Δθ = np.array([np.cos(θ), np.sin(θ)])
                satellite_center = self.planets[planet_name].center + satellite.orbit_r * Δθ
                satellite_k = plt.Circle(satellite_center, satellite.radius, color="green", alpha=1)
                ax.add_patch(satellite_k)
                satellite_k = plt.Circle(
                    satellite_center, satellite.radius + kappa_sats[name][k], color="red", alpha=0.2
                )
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
                ax.arrow(
                    X[0, k] - self.sg.l_r * np.cos(X[2, k]),
                    X[1, k] - self.sg.l_r * np.sin(X[2, k]),
                    self.sg.l * np.cos(X[2, k]),
                    self.sg.l * np.sin(X[2, k]),
                    width=0.05,
                    length_includes_head=True,
                )
                ax.arrow(
                    X[0, k] - self.sg.l_r * np.cos(X[2, k]),
                    X[1, k] - self.sg.l_r * np.sin(X[2, k]),
                    U[0, k] * np.cos(U[1, k] + X[2, k]) / 2,
                    U[0, k] * np.sin(U[1, k] + X[2, k]) / 2,
                    width=0.03,
                    color="r",
                    length_includes_head=True,
                )
                ax.arrow(
                    X[0, k + 1] - self.sg.l_r * np.cos(X[2, k + 1]),
                    X[1, k + 1] - self.sg.l_r * np.sin(X[2, k + 1]),
                    self.sg.l * np.cos(X[2, k + 1]),
                    self.sg.l * np.sin(X[2, k + 1]),
                    width=0.05,
                    length_includes_head=True,
                )
                ax.arrow(
                    X[0, k + 1] - self.sg.l_r * np.cos(X[2, k + 1]),
                    X[1, k + 1] - self.sg.l_r * np.sin(X[2, k + 1]),
                    U[0, k + 1] * np.cos(U[1, k + 1] + X[2, k + 1]) / 2,
                    U[0, k + 1] * np.sin(U[1, k + 1] + X[2, k + 1]) / 2,
                    width=0.03,
                    color="r",
                    length_includes_head=True,
                )

                block = plt.Circle(X[0:2, k], min_kappa[k], alpha=0.2, color="blue")
                ax.add_patch(block)
                block = plt.Circle(
                    X[0:2, k + 1],
                    min_kappa[k],
                    alpha=0.2,
                    color="blue",
                )
                ax.add_patch(block)

                savedir = (
                    "../../out/11/"
                    + str(len(self.satellites))
                    + "_"
                    + str(round(self.planets["Namek"].center[1], 2))
                    + "/vis_k"
                )
                if not os.path.exists(savedir):
                    os.mkdir(savedir)
                fig.savefig(
                    savedir + "/vis_" + str(iteration) + "_" + str(k) + ".png",
                    bbox_inches="tight",
                )
                plt.close(fig)

    def vis_sp(self, center, corners, dock_targets):
        x, y, psi = center
        # psi = psi
        fig, axes = plt.subplots()
        axes.scatter(x, y, marker="x")
        for i, (l, theta) in enumerate(corners):
            x_c = x + l * np.cos(psi + theta)
            y_c = y + l * np.sin(psi + theta)
            axes.scatter(x_c, y_c, label=str(i))
            # for t in dock_targets:
            # print(i, np.linalg.norm(np.array([x_c, y_c]) - np.array(t), 2))
        axes.legend()
        axes.set_aspect("equal", adjustable="box")
        if dock_targets is not None:
            axes.plot([x[0] for x in dock_targets], [x[1] for x in dock_targets])

        savedir = "../../out/11/" + str(len(self.satellites)) + "_" + str(round(self.planets["Namek"].center[1], 2))
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        fig.savefig(
            savedir + "/sp_recon.png",
            bbox_inches="tight",
        )
        plt.close(fig)
