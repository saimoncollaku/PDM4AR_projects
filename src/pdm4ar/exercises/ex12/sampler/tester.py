from pdm4ar.exercises.ex12.sampler.frenet_sampler import FrenetSampler
from pdm4ar.exercises.ex12.sampler.b_spline import SplineReference
import matplotlib.pyplot as plt
import numpy as np
import random


def main():
    # Reference path
    num_points = 10
    radius = 20
    theta = np.linspace(0, np.pi, num_points, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # * ****************************************************************
    # 1- create reference path, spline is needed to add "continuity" (resolution HAS TO BE BIG ENOUGH)
    reference_points = np.column_stack((x, y))
    spline_ref = SplineReference(reference_points, resolution=1000)
    ref_points = np.column_stack((spline_ref.x, spline_ref.y))
    ref_frenet = spline_ref.to_frenet(ref_points)
    # 2- add initial condition, they have to be calculated, fetched from the car state
    c_speed = 20 / 3.6  # current speed [m/s]
    c_d = 2.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current latral acceleration [m/s]
    s0 = 0.0  # current course position
    # 3- define the sampler
    sampler = FrenetSampler(1, 50, 5, 5, 1, c_speed, c_d, c_d_d, c_d_dd, s0)
    # 4-  create array of path samples
    fp = sampler.get_paths()
    # 5- get the best fp index (here i assigned 5)
    frenet_points = np.column_stack((fp[5].s, fp[5].d))
    path_points = spline_ref.to_cartesian(frenet_points)
    # 6- assign the initial conditions from 2 given the best path
    # * ****************************************************************

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original vs Reconstructed Paths")
    plt.plot(ref_points[:, 0], ref_points[:, 1], "b-", label="Reference Trajectory")
    for i, path in enumerate(fp):
        frenet_points = np.column_stack((path.s, path.d))
        path_points = spline_ref.to_cartesian(frenet_points)
        if random.random() < 0.01:
            plt.scatter(path_points[:, 0], path_points[:, 1], label=f"Path {i}")
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.legend()
    plt.axis("equal")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title("Frenet Frame Transformation")
    plt.plot(ref_frenet[:, 0], ref_frenet[:, 1], "b-", label="Reference Trajectory")
    for i, path in enumerate(fp):
        frenet_points = np.column_stack((path.s, path.d))
        if random.random() < 0.01:
            plt.scatter(frenet_points[:, 0], frenet_points[:, 1], label=f"Path {i}")

    plt.xlabel("s (Arc Length)")
    plt.ylabel("d (Lateral Distance)")
    # plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("frenet.png")

    # plt.figure(figsize=(12, 5))

    # plt.subplot(1, 2, 1)
    # plt.title("Original vs Reconstructed Paths")
    # plt.plot(ref_points[:, 0], ref_points[:, 1], "b-", label="Reference Trajectory")
    # plt.scatter(path_points[:, 0], path_points[:, 1], color="r")
    # # plt.plot(path_points2[:, 0], path_points2[:, 1], "g-")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.legend()
    # plt.axis("equal")
    # plt.grid(True)

    # # Frenet Transformation
    # plt.subplot(1, 2, 2)
    # plt.title("Frenet Frame Transformation")
    # plt.plot(ref_frenet[:, 0], ref_frenet[:, 1], "b-", label="Reference Trajectory")
    # plt.scatter(frenet_points[:, 0], frenet_points[:, 1], color="r")
    # recon = spline_ref.to_frenet(path_points)
    # plt.scatter(recon[:, 0], recon[:, 1], color="g")

    # # plt.scatter(frenet_points2[:, 0], frenet_points2[:, 1], color="g")
    # plt.xlabel("s (Arc Length)")
    # plt.ylabel("d (Lateral Distance)")
    # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.savefig("frenet.png")


if __name__ == "__main__":
    main()


# plt.figure(figsize=(8, 6))
# sampler = FrenetSampler(30 / 3.6, 2, 3, 1, c_speed, c_d, c_d_d, c_d_dd, s0)
# fp = sampler.calc_frenet_paths()
# plt.plot(fp[5].s, fp[5].d, "ro")
# sampler.assign_last_cond(5)
# fp = sampler.calc_frenet_paths()
# plt.plot(fp[60].s, fp[60].d, "ro")

# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()
# plt.grid(True)
# plt.savefig("frenet.png")
