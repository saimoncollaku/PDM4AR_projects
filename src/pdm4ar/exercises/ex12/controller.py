import numpy as np
from matplotlib import pyplot as plt
import casadi as ca


class MPController:
    def __init__(self, vehicle_params, vehicle_geom):
        self.target_traj = None
        self.sp = vehicle_params
        self.sg = vehicle_geom

        self.N = 5
        self.dt = 0.1

        self.Q = np.diag([1.0, 1.0, 0.1, 1.0, 0.1])
        self.R = np.diag([0.01, 0.01])

        self.opti = ca.Opti("conic")

        self.X = self.opti.variable(5, self.N + 1)
        self.U = self.opti.variable(2, self.N)
        self.X_ref = self.opti.parameter(5, self.N + 1)

        cost = 0

        for k in range(self.N):
            state = self.X[:, k]
            control = self.U[:, k]
            next_state = self.X[:, k + 1]

            ref_error = state - self.X_ref[:, k]

            psidot = state[3] * ca.tan(state[4]) / self.sg.wheelbase
            xdot = state[3] * ca.cos(state[2]) - (psidot * self.sg.lr * ca.sin(state[2]))
            ydot = state[3] * ca.sin(state[2]) + (psidot * self.sg.lr * ca.cos(state[2]))
            vxdot = control[0]
            deltadot = control[1]

            self.opti.subject_to(state[0] + self.dt * xdot == next_state[0])
            self.opti.subject_to(state[1] + self.dt * ydot == next_state[1])
            self.opti.subject_to(state[2] + self.dt * psidot == next_state[2])
            self.opti.subject_to(state[3] + self.dt * vxdot == next_state[3])
            self.opti.subject_to(state[4] + self.dt * deltadot == next_state[4])

            cost += control.T @ self.R @ control
            cost += ref_error.T @ self.Q @ ref_error

        self.opti.subject_to(self.X[2, :] <= np.pi)
        self.opti.subject_to(self.X[2, :] >= -np.pi)
        self.opti.subject_to(self.X[3, :] <= 25)
        self.opti.subject_to(self.X[3, :] >= 5)
        self.opti.subject_to(self.X[4, :] <= self.sp.delta_max)
        self.opti.subject_to(self.X[4, :] >= -self.sp.delta_max)
        self.opti.subject_to(self.U[0, :] <= self.sp.acc_limits[1])
        self.opti.subject_to(self.U[0, :] >= self.sp.acc_limits[0])
        self.opti.subject_to(self.U[1, :] <= self.sp.ddelta_max)
        self.opti.subject_to(self.U[1, :] >= -self.sp.ddelta_max)

        self.cost = cost
        self.opti.minimize(cost)
        self.solver = self.opti.solver("qpoases")

        self.curr_idx = -1
        self.fig, self.axes = plt.subplots(2, 1)
        self.curr_states = []

    def plot_controller_perf(self, t):
        target_states = [[state.vx, state.psi] for state in self.target_traj.values]

        self.axes[0].plot(self.target_traj.timestamps, [v[0] for v in target_states], c="r")
        self.axes[0].plot(
            self.target_traj.timestamps[: len(self.curr_states)], [v[0] for v in self.curr_states], c="b", label="v"
        )
        # self.axes[0].plot(self.target_traj.timestamps[: len(self.curr_states)], self.error_v, c="k")

        self.axes[1].plot(self.target_traj.timestamps, [v[1] for v in target_states], c="r")
        self.axes[1].plot(
            self.target_traj.timestamps[: len(self.curr_states)], [v[1] for v in self.curr_states], c="b", label="psi"
        )
        self.axes[1].plot(self.target_traj.timestamps[self.curr_idx], target_states[self.curr_idx][1], marker="x")
        # self.axes[1].plot(self.target_traj.timestamps[: len(self.curr_states)], self.error_psi, c="k")
        self.fig.savefig("../../out/12/controller_perf" + str(round(float(t), 2)) + ".png")

    def set_reference(self, target):
        self.target_traj = target
        self.x_ref = np.array(
            [[state.x, state.y, state.psi, state.vx, state.delta] for state in self.target_traj.values]
        )
        plt.close(self.fig)
        self.curr_idx = 0
        self.fig, self.axes = plt.subplots(2, 1)
        self.curr_states = []

    def update_progress(self, curr_state):
        dx = [state.x - curr_state.x for state in self.target_traj.values]
        dy = [state.y - curr_state.y for state in self.target_traj.values]
        return np.argmin(np.hypot(dx, dy))

    def solve(self, curr_state, x_ref):
        self.opti.set_value(self.X_ref, ca.DM(x_ref.T))
        self.opti.set_initial(
            self.X,
            ca.repmat(
                ca.DM([curr_state.x, curr_state.y, curr_state.psi, curr_state.vx, curr_state.delta]), 1, self.N + 1
            ),
        )
        self.opti.set_initial(self.U, ca.DM.zeros(2, self.N))
        sol = self.opti.solve()
        controls = sol.value(self.U)
        print(sol.value(self.cost))
        return controls[0, :], controls[1, :]

    def get_controls(self, curr_state, t):
        self.curr_states.append([curr_state.vx, curr_state.psi])
        print(self.curr_idx)
        if self.curr_idx + self.N < self.x_ref.shape[0]:
            self.mpc_sol = self.solve(curr_state, self.x_ref[self.curr_idx : self.curr_idx + self.N + 1])
            acc = self.mpc_sol[0][0]
            ddelta = self.mpc_sol[1][0]
        else:
            idx = self.x_ref.shape[0] - self.curr_idx
            acc = self.mpc_sol[0][idx]
            ddelta = self.mpc_sol[1][idx]

        self.curr_idx = self.update_progress(curr_state)
        return acc, ddelta


class BasicController:

    def __init__(self, vehicle_params, vehicle_geom, visualize):
        self.target_traj = None
        self.sp = vehicle_params
        self.sg = vehicle_geom
        self.p_gain = 8.0
        self.i_gain = 6.0
        self.d_gain = 2.0

        self.p_gain_delta = 10.0
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

        self.visualize = visualize
        if self.visualize:
            self.fig, self.axes = plt.subplots(2, 1)

    def plot_controller_perf(self, t):
        target_states = [[state.vx, state.delta] for state in self.target_traj.values]

        self.axes[0].plot(self.target_traj.timestamps, [v[0] for v in target_states], c="r")
        self.axes[0].plot(
            self.target_traj.timestamps[: len(self.curr_states)], [v[0] for v in self.curr_states], c="b", label="v"
        )
        self.axes[0].plot(
            [self.target_traj.timestamps[self.curr_idx], self.target_traj.timestamps[self.curr_idx]],
            [target_states[self.curr_idx][0], self.curr_states[-1][0]],
            marker="x",
        )
        # self.axes[0].plot(self.target_traj.timestamps[: len(self.curr_states)], self.error_v, c="k")

        self.axes[1].plot(self.target_traj.timestamps, [v[1] for v in target_states], c="r")
        self.axes[1].plot(
            self.target_traj.timestamps[: len(self.curr_states)], [v[1] for v in self.curr_states], c="b", label="psi"
        )
        self.axes[1].plot(
            [self.target_traj.timestamps[self.curr_idx], self.target_traj.timestamps[self.curr_idx]],
            [target_states[self.curr_idx][1], self.curr_states[-1][1]],
            marker="x",
        )
        # if len(self.curr_states) > 2:
        #     print(
        #         max(np.gradient([state[1] for state in target_states])),
        #         max(np.gradient([state[1] for state in self.curr_states])),
        #     )
        # self.axes[1].plot(self.target_traj.timestamps[: len(self.curr_states)], self.error_psi, c="k")
        self.axes[0].autoscale()
        self.axes[1].autoscale()
        self.fig.savefig("../../out/12/controller_perf" + str(round(float(t), 2)) + ".png")

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
        # ddelta = 0
        self.curr_idx = self.update_progress_time(float(t))
        # print("curr_state: ", curr_state.vx, curr_state.psi)
        # print(acc, ddelta)
        return acc, ddelta
