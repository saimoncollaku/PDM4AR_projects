import logging

from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.sim_types import SimTime
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters

from dg_commons import PlayerName
from pdm4ar.exercises.ex12.planner import Planner

from pdm4ar.exercises.ex12.params import Pdm4arAgentParams

logger = logging.getLogger(__name__)
logging.basicConfig(encoding="utf-8", level=logging.WARNING, format="%(levelname)s %(name)s:\t%(message)s")


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    name: PlayerName
    sg: VehicleGeometry
    sp: VehicleParameters
    planner: Planner

    all_timesteps: list[SimTime]
    all_states: list[VehicleState]

    def __init__(self):
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()
        self.planner = None

    def on_episode_init(self, init_obs: InitSimObservations):
        """
        This method is called by the simulator only at the beginning of each simulation.
        Do not modify the signature of this method.
        """
        self.name = init_obs.my_name
        logger.warning("Starting new scenario")
        assert isinstance(init_obs.model_geometry, VehicleGeometry)
        assert isinstance(init_obs.model_params, VehicleParameters)
        self.sg = init_obs.model_geometry
        self.sp = init_obs.model_params

        self.planner = Planner(init_obs, self.sp, self.sg, self.params)

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """
        assert isinstance(self.planner, Planner)
        cmd_acc, cmd_ddelta = self.planner.get_commands(sim_obs)
        return VehicleCommands(cmd_acc, cmd_ddelta)
