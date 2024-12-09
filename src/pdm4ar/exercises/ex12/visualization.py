from typing import Optional, Sequence, Union

from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PathCollection

import numpy as np

from dg_commons import Color
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.planning.trajectory import Trajectory
from dg_commons.sim.simulator_structures import InitSimObservations, SimObservations

from commonroad.visualization.draw_params import MPDrawParams
from commonroad.visualization.mp_renderer import MPRenderer

from pdm4ar.exercises_def.ex12.sim_context import ZOrders


class Visualizer:

    def __init__(self, init_obs: InitSimObservations):
        self.dg_scenario = init_obs.dg_scenario
        self.fig, self.axes = plt.subplots(figsize=(10, 10))

        self.commonroad_renderer: MPRenderer = MPRenderer(ax=self.axes, draw_params=MPDrawParams())
        self.shapely_viz = ShapelyViz(ax=self.commonroad_renderer.ax)
        self.draw_params = self.commonroad_renderer.draw_params

        self.missions = {}
        self.colors = {}

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
    ):
        segments, mcolor = [], []
        ax = self.shapely_viz.ax
        for traj in trajectories:
            sampled_traj = np.vstack([[x.x, x.y, x.vx] for x in traj.values])
            segments.append(sampled_traj[:, :2])
            # mcolor.append(sampled_traj[:, 2])  # fixme marker color functionality not available yet

        if traj_lines is None:
            traj_lines = LineCollection(
                segments=[], colors=colors, linewidths=width, alpha=alpha, zorder=ZOrders.TRAJECTORY
            )
            size = np.linalg.norm(ax.bbox.size) / 1000
            traj_points = ax.scatter([], [], alpha=0, s=size, c="r", zorder=ZOrders.TRAJECTORY_MARKER)

        assert traj_lines is not None
        traj_lines.set_segments(segments=segments)
        traj_lines.set_color(colors)
        traj_points.set_offsets(np.concatenate(segments))
        # traj_points.set_facecolor(mcolor)  # todo adjust color based on velocity

        ax.add_collection(traj_lines)
        ax.add_collection(traj_points)

        # # https://stackoverflow.com/questions/23966121/updating-the-positions-and-colors-of-pyplot-scatter
        # return traj_lines, traj_points

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
