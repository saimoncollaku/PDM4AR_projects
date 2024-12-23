from typing import Optional, Sequence, Union

from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PathCollection

import numpy as np

from dg_commons import Color
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.planning.trajectory import Trajectory
from dg_commons.sim.simulator_structures import InitSimObservations, SimObservations
from dg_commons.sim.models.vehicle import VehicleState

from commonroad.visualization.draw_params import MPDrawParams
from commonroad.visualization.mp_renderer import MPRenderer

from pdm4ar.exercises.ex12.sampler.sample import Sample, Samplers
from pdm4ar.exercises.ex12.sampler.dubins_algo import Dubins
from pdm4ar.exercises_def.ex12.sim_context import ZOrders
from dg_commons.planning import Trajectory


class Visualizer:

    def __init__(self, init_obs: InitSimObservations):
        self.dg_scenario = init_obs.dg_scenario
        self.fig, self.axes = plt.subplots(figsize=(20, 20))

        self.commonroad_renderer: MPRenderer = MPRenderer(ax=self.axes, draw_params=MPDrawParams())
        self.shapely_viz = ShapelyViz(ax=self.commonroad_renderer.ax)
        self.draw_params = self.commonroad_renderer.draw_params

        self.missions = {}
        self.colors = {}

    def clear_viz(self):
        plt.close(self.fig)
        self.fig, self.axes = plt.subplots(figsize=(20, 20))
        self.commonroad_renderer.ax = self.axes
        self.shapely_viz.ax = self.axes

    def set_goal(self, agent_name, goal, model):
        self.missions[agent_name] = goal
        self.colors[agent_name] = model.color

    def save_fig(self, savepath="../../out/12/scene.png"):
        self.fig.savefig(savepath, bbox_inches="tight")

    def plot_trajectories(
        self,
        trajectories: Sequence[Trajectory],
        traj_lines: Optional[LineCollection] = None,
        traj_points: Optional[PathCollection] = None,
        colors: Union[list[Color], Color] = ["black", "firebrick"],
        width: float = 1.5,
        alpha: float = 1,
        plot_samples=False,
        plot_heading=False,
    ):
        segments, mcolor = [], []
        ax = self.shapely_viz.ax
        for traj in trajectories:
            sampled_traj = np.vstack([[x.x, x.y, x.psi] for x in traj.values])
            segments.append(sampled_traj[:, :2])
            # mcolor.append(sampled_traj[:, 2])  # fixme marker color functionality not available yet

        if traj_lines is None:
            traj_lines = LineCollection(
                segments=[], colors=colors, linewidths=width, alpha=alpha, zorder=ZOrders.TRAJECTORY
            )
            size = np.linalg.norm(ax.bbox.size) / 1000
            traj_points = ax.scatter(
                [], [], alpha=0.4 if plot_samples else 0.0, s=size, c="k", zorder=ZOrders.TRAJECTORY_MARKER
            )

        assert traj_lines is not None
        traj_lines.set_segments(segments=segments)
        traj_lines.set_color(colors)
        traj_points.set_offsets(np.concatenate(segments))
        # traj_points.set_facecolor(mcolor)  # todo adjust color based on velocity

        ax.add_collection(traj_lines)
        ax.add_collection(traj_points)

        if plot_heading:
            x_mid = []
            y_mid = []
            dx = []
            dy = []

            for traj in trajectories:
                pts = traj.values
                for i in range(len(pts) - 1):
                    start, end = np.array([pts[i].x, pts[i].y]), np.array([pts[i + 1].x, pts[i + 1].y])
                    mid_point = (start + end) / 2  # Midpoint of the segment
                    direction = end - start  # Vector of the segment
                    norm = np.linalg.norm(direction)
                    if norm > 0:  # Normalize direction
                        direction /= norm

                    # Append midpoint and direction
                    x_mid.append(mid_point[0])
                    y_mid.append(mid_point[1])
                    dx.append(norm * np.cos(pts[i].psi))
                    dy.append(norm * np.sin(pts[i].psi))

            # Use quiver to plot arrows
            ax.quiver(x_mid, y_mid, dx, dy, angles="xy", scale_units="xy", scale=1, color="red", width=0.005)

        # # https://stackoverflow.com/questions/23966121/updating-the-positions-and-colors-of-pyplot-scatter
        # return traj_lines, traj_points

    def plot_reference(self, reference: np.ndarray):
        self.axes.scatter(reference[:, 0], reference[:, 1], c="red", marker="x")
        self.save_fig("../../out/12/reference.png")

    def plot_scenario(self, obs: SimObservations):
        if self.dg_scenario.scenario:
            self.draw_params.lanelet_network.traffic_light.draw_traffic_lights = True
            self.dg_scenario.lanelet_network.draw(self.commonroad_renderer, draw_params=self.draw_params)
            self.commonroad_renderer.render()

        for pn, goal in self.missions.items():
            self.shapely_viz.add_shape(
                goal.get_plottable_geometry(),
                color=self.colors[pn],
                zorder=ZOrders.GOAL,
                alpha=0.5,
            )

        for pn, model in obs.players.items():
            footprint = model.occupancy
            color = "firebrick" if pn in self.missions else "royalblue"
            self.shapely_viz.add_shape(footprint, color=color, zorder=ZOrders.MODEL, alpha=0.5)

        ax = self.shapely_viz.ax
        ax.autoscale()
        ax.set_aspect("equal")

    def plot_samples(self, samples, wb, num_plan):
        if samples is None:
            return
        feas_trajectories = []
        feas_idx = [idx for idx, sample in enumerate(samples) if sample.cost != np.inf]
        if len(feas_idx) > 0:
            for s_idx in np.random.choice(feas_idx, 50):
                path: Sample = samples[s_idx]
                path.compute_steering(wb)
                timestamps = list(path.t)
                states = [
                    VehicleState(path.x[i], path.y[i], path.psi[i], path.vx[i], path.delta[i]) for i in range(path.T)
                ]
                feas_trajectories.append(Trajectory(timestamps, states))
            self.plot_trajectories(feas_trajectories, colors=["royalblue" for traj in feas_trajectories])

        all_trajectories = []
        for s_idx in np.random.choice(range(len(samples)), 50):
            path = samples[s_idx]
            path.compute_steering(wb)
            timestamps = list(path.t)
            states = [VehicleState(path.x[i], path.y[i], path.psi[i], path.vx[i], path.delta[i]) for i in range(path.T)]
            all_trajectories.append(Trajectory(timestamps, states))
        self.plot_trajectories(all_trajectories, colors=["grey" for traj in all_trajectories], alpha=0.2)
        self.save_fig(f"../../out/12/samples_{type_planner}.png")

    def plot_samples_without_background(self, best_sample: Trajectory, samples: list[Sample]):
        start_state = [best_sample._values[0].x, best_sample._values[0].y, best_sample._values[0].psi]
        end_state = [best_sample._values[-1].x, best_sample._values[-1].y, best_sample._values[-1].psi]
        x = [state.x for state in best_sample._values]
        y = [state.y for state in best_sample._values]
        psi = [state.psi for state in best_sample._values]
        Dubins.plot_any_trajectory(
            x,
            y,
            psi,
            start_state,
            end_state,
            "../../out/12/all_samples_only.png",
            samples,
        )
