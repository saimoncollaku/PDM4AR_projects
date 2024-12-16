from hypothesis import target
import numpy as np


class BasicController:

    def __init__(self, vehicle_params, vehicle_geom):
        self.target_traj = None
        self.sp = vehicle_params
        self.sg = vehicle_geom
        self.p_gain = 1.0
        self.psi_gain = 0.6
        self.k_dd = 0.5
        self.dt = 0.1
        self.l = self.sg.lf
        self.curr_idx = -1

    def set_reference(self, target):
        self.target_traj = target
        self.curr_idx = 0

    def update_progress(self, curr_state):
        dx = [state.x - curr_state.x for state in self.target_traj.values]
        dy = [state.y - curr_state.y for state in self.target_traj.values]
        return np.argmin(np.hypot(dx, dy))

    def speed_control(self, curr_vx):
        target = self.target_traj.values[self.curr_idx].vx
        acc = self.p_gain * (target - curr_vx)
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
        return min(self.curr_idx + 2, len(self.target_traj.values) - 1)

    def steer_control(self, curr_state):
        target = self.target_traj.values[self.curr_idx].psi
        return self.psi_gain * (target - curr_state.psi)
        # l_idx = self.lookahead_idx(self.curr_idx, curr_state)
        # t_setpt = self.target_traj.values[l_idx]

        # alpha = np.arctan2(t_setpt.y - curr_state.y + self.l * np.sin(curr_state.psi), t_setpt.x - curr_state.x + self.l * np.cos(curr_state.psi))

        # delta = np.arctan2(2.0 * )

    # def steer_control(self, curr_state):
    def stanley_steer_control(self, curr_state):
        # Get the current target point
        l_idx = self.lookahead_idx(self.curr_idx, curr_state)
        target = self.target_traj.values[l_idx]

        # Compute heading error (wrap to [-pi, pi] for continuity)
        psi_e = target.psi - curr_state.psi

        # Compute the lateral error (distance perpendicular to front axle direction)
        front_axle_vec = [
            np.cos(curr_state.psi + np.pi / 2),
            np.sin(curr_state.psi + np.pi / 2),
        ]  # Align with current heading
        dx = target.x - curr_state.x - self.sg.lf * np.cos(curr_state.psi)
        dy = target.y - curr_state.y - self.sg.lf * np.sin(curr_state.psi)
        axle_error = np.dot([dx, dy], front_axle_vec)

        # Compute desired steering angle to correct lateral error
        psi_d = np.arctan2(self.psi_gain * axle_error, curr_state.vx + 1e-3)  # Avoid division by zero

        steering_angle = np.clip(psi_e + psi_d, -self.sp.ddelta_max, self.sp.ddelta_max)
        return steering_angle

    def get_controls(self, curr_state, t):
        acc = self.speed_control(curr_state.vx)
        ddelta = self.stanley_steer_control(curr_state)
        # print("curr_state: ", curr_state.vx, curr_state.psi)
        # print(acc, ddelta)
        return acc, ddelta
