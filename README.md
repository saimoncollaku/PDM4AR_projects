## Problem Request & Approach
**Request:**  
Implement a planning and control stack that receives observations from a 2D Lidar sensor and outputs control commands to execute a lane change. The system must navigate mixed traffic without collisions, while respecting vehicle dynamics and road constraints.

**Approach:**  
Our implementation is **sampling-based** using **Frenet curves**. This method allowed us to:
- **Sample multiple candidate trajectories:** by converting the problem into the Frenet coordinate system.
- **Evaluate each trajectory:** based on safety, efficiency, and comfort, considering the vehicle's kinematic constraints.
- **Select the optimal path:** that facilitates a smooth and safe lane change.
- **Integrate with the simulator:** ensuring timely command updates in a closed-loop setup.

## Demonstration
Below is a video demonstration of our highway driving solution. Click on the preview to play the video.

<div align="center">
  <a href="car.mp4">
    <img src="car.gif" alt="Highway Driving Demonstration" style="max-width:600px;">
  </a>
</div>
