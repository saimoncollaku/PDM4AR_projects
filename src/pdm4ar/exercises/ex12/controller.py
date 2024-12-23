import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


class BasicController:

    def __init__(self, vehicle_params, vehicle_geom, visualize):
        self.target_traj = None
        self.sp = vehicle_params
        self.sg = vehicle_geom
        self.p_gain = 8.0
        self.i_gain = 6.0
        self.d_gain = 2.0

        self.p_gain_delta = 8.0
        self.i_gain_delta = 0.5
        self.d_gain_delta = 4.0

        self.psi_p_gain = 1.6
        self.psi_l_gain = 0.4
        self.psi_d_gain = 0.8
        self.psi_i_gain = 0.02

        self.k_dd = 0.5
        self.dt = 0.1
        self.l = self.sg.lf
        self.curr_idx = -1
        self.curr_states = []
        self.track_error = 0

        self.visualize = visualize
        if self.visualize:
            self.fig, self.axes = plt.subplots(2, 1)

    def plot_controller_perf(self, t):
        if self.target_traj is None:
            return

        target_states = [[state.vx, state.delta] for state in self.target_traj.values]

        self.axes[0].plot(self.target_traj.timestamps, [v[0] for v in target_states], c="r")
        self.axes[0].plot(self.target_traj.timestamps[: len(self.curr_states)], [v[0] for v in self.curr_states], c="b")
        self.axes[0].plot(
            [self.target_traj.timestamps[self.curr_idx], self.target_traj.timestamps[self.curr_idx]],
            [target_states[self.curr_idx][0], self.curr_states[-1][0]],
            marker="x",
        )
        self.axes[0].set_ylabel("vx")
        # self.axes[0].plot(self.target_traj.timestamps[: len(self.curr_states)], self.error_v, c="k")

        self.axes[1].plot(self.target_traj.timestamps, [v[1] for v in target_states], c="r")
        self.axes[1].plot(self.target_traj.timestamps[: len(self.curr_states)], [v[1] for v in self.curr_states], c="b")
        self.axes[1].plot(
            [self.target_traj.timestamps[self.curr_idx], self.target_traj.timestamps[self.curr_idx]],
            [target_states[self.curr_idx][1], self.curr_states[-1][1]],
            marker="x",
        )
        self.axes[1].set_ylabel("delta")

        patchList = []
        legend_dict = {"trajectory": "red", "tracking": "blue"}
        for key, color in legend_dict.items():
            data_key = mpatches.Patch(color=color, label=key)
            patchList.append(data_key)

        self.axes[0].legend(handles=patchList)
        self.axes[1].legend(handles=patchList)
        # if len(self.curr_states) > 2:
        #     print(
        #         max(np.gradient([state[1] for state in target_states])),
        #         max(np.gradient([state[1] for state in self.curr_states])),
        #     )
        # self.axes[1].plot(self.target_traj.timestamps[: len(self.curr_states)], self.error_psi, c="k")
        self.axes[0].autoscale()
        self.axes[1].autoscale()
        # self.fig.savefig("../../out/12/controller_perf" + str(round(float(t), 2)) + ".png")
        self.fig.savefig("../../out/12/controller_perf.png")

    def clear_viz(self):
        plt.close(self.fig)

    def set_reference(self, target):
        self.target_traj = target
        self.curr_idx = 0
        self.curr_states = []
        self.error_v = []
        self.error_psi = []
        if self.visualize:
            self.fig, self.axes = plt.subplots(2, 1)

    def update_progress(self, curr_state):
        dx = [state.x - curr_state.x for state in self.target_traj.values]
        dy = [state.y - curr_state.y for state in self.target_traj.values]
        return np.argmin(np.hypot(dx, dy))

    def update_progress_time(self, t):
        dt = [abs(ts - t) for ts in self.target_traj.timestamps]
        return np.argmin(dt)

    def speed_control(self, curr_vx):
        l_idx = min(self.curr_idx + 1, len(self.target_traj.values) - 1)
        target = self.target_traj.values[l_idx].vx
        # print(target, curr_vx, self.sp.vx_limits[1])
        self.error_v.append(target - curr_vx)
        if len(self.error_v) > 1:
            d_error = self.error_v[-1] - self.error_v[-2]
        else:
            d_error = 0
        i_error = sum(self.error_v)
        acc = self.p_gain * (target - curr_vx) + self.d_gain * d_error + self.i_gain * i_error
        acc = np.clip(acc, self.sp.acc_limits[0], self.sp.acc_limits[1])
        return acc

    def lookahead_idx(self, idx, curr_state):
        # target_idx = idx
        # t_setpt = self.target_traj.values[target_idx]
        # la_dist = self.k_dd * curr_state.vx
        # x = curr_state.x - self.sg.lr * np.cos(curr_state.psi)
        # y = curr_state.y - self.sg.lr * np.sin(curr_state.psi)
        # while np.linalg.norm([x - t_setpt.x, y - t_setpt.y], 2) < la_dist:
        #     if target_idx == len(self.target_traj.values) - 1:
        #         break
        #     target_idx += 1
        #     t_setpt = self.target_traj.values[target_idx]
        # return target_idx
        return min(self.curr_idx + 5, len(self.target_traj.values) - 1)

    def steer_control(self, curr_state):
        l_idx = min(self.curr_idx + 1, len(self.target_traj.values) - 1)
        target = self.target_traj.values[l_idx]
        # print(target, curr_vx, self.sp.vx_limits[1])
        psi_e = target.delta - curr_state.delta
        self.error_psi.append(psi_e)
        if len(self.error_psi) > 1:
            d_error = self.error_psi[-1] - self.error_psi[-2]
        else:
            d_error = 0
        i_error = sum(self.error_psi)
        steering_angle = self.p_gain_delta * (psi_e) + self.d_gain_delta * d_error + self.i_gain_delta * i_error
        steering_angle = np.clip(steering_angle, -self.sp.ddelta_max, self.sp.ddelta_max)
        # print(psi_e, target.delta, curr_state.delta, steering_angle, self.sp.ddelta_max)
        return steering_angle

    # def steer_control(self, curr_state):
    def stanley_steer_control(self, curr_state):
        # Get the current target point
        # l_idx = self.lookahead_idx(self.curr_idx, curr_state)
        target = self.target_traj.values[self.curr_idx]

        # Compute heading error (wrap to [-pi, pi] for continuity)
        psi_e = (target.psi - curr_state.psi + np.pi) % (2 * np.pi) - np.pi
        self.error_psi.append(psi_e)
        if len(self.error_psi) > 1:
            d_error = self.error_psi[-1] - self.error_psi[-2]
        else:
            d_error = 0
        # Compute the lateral error (distance perpendicular to front axle direction)
        front_axle_vec = [
            np.cos(curr_state.psi - np.pi / 2),
            np.sin(curr_state.psi - np.pi / 2),
        ]  # Align with current heading
        dx = target.x - curr_state.x
        dy = target.y - curr_state.y
        axle_error = np.dot([dx, dy], front_axle_vec)

        i_error = sum(self.error_psi)
        # Compute desired steering angle to correct lateral error
        psi_d = np.arctan2(self.psi_l_gain * axle_error, curr_state.vx + 1e-3)  # Avoid division by zero
        steering_angle = self.psi_p_gain * psi_e + self.psi_d_gain * d_error + psi_d + self.psi_i_gain * i_error

        steering_angle = np.clip(steering_angle, -self.sp.ddelta_max, self.sp.ddelta_max)
        print(psi_e, target.psi, curr_state.psi, steering_angle, self.sp.ddelta_max)
        return steering_angle

    def get_controls(self, curr_state, t):
        self.curr_states.append([curr_state.vx, curr_state.delta])
        acc = self.speed_control(curr_state.vx)
        ddelta = self.steer_control(curr_state)
        self.track_error = np.linalg.norm(
            np.array([curr_state.x, curr_state.y])
            - np.array([self.target_traj.values[self.curr_idx].x, self.target_traj.values[self.curr_idx].y]),
            ord=2,
        )
        self.curr_idx = self.update_progress_time(float(t))
        return acc, ddelta
