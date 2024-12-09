# Generating the plan for the best trajectory > Execution and Tracking, Replaning


class Planner:

    def __init__(self) -> None:
        pass

    def _check_if_need_replan(self):
        pass

    # scenario update once
    def plan(self) -> None:
        if not self._check_if_need_replan():
            return None  # something
        # trajectory sampling
        # trajectory evalulation
        # choose the best trajectory and return
        pass
