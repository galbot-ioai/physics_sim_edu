######################################################################################
# Copyright (c) 2023-2025 Galbot. All Rights Reserved.
#
# This software contains confidential and proprietary information of Galbot, Inc.
# ("Confidential Information"). You shall not disclose such Confidential Information
# and shall use it only in accordance with the terms of the license agreement you
# entered into with Galbot, Inc.
#
# UNAUTHORIZED COPYING, USE, OR DISTRIBUTION OF THIS SOFTWARE, OR ANY PORTION OR
# DERIVATIVE THEREOF, IS STRICTLY PROHIBITED. IF YOU HAVE RECEIVED THIS SOFTWARE IN
# ERROR, PLEASE NOTIFY GALBOT, INC. IMMEDIATELY AND DELETE IT FROM YOUR SYSTEM.
######################################################################################
#
#  ██████  ██    ██ ██    ██ ████████ ██     ██ ██    ██  ███████  ██     ██    ███
# ██    ██  ██  ██  ███   ██    ██    ██     ██ ███   ██ ██     ██ ██     ██   ██ ██
# ██         ████   ████  ██    ██    ██     ██ ████  ██ ██     ██ ██     ██  ██   ██
#  ██████     ██    ██ ██ ██    ██    █████████ ██ ██ ██ ██     ██ ██     ██ ██     ██
#       ██    ██    ██  ████    ██    ██     ██ ██  ████ ██     ██  ██   ██  █████████
# ██    ██    ██    ██   ███    ██    ██     ██ ██   ███ ██     ██   ██ ██   ██     ██
#  ██████     ██    ██    ██    ██    ██     ██ ██    ██  ███████     ███    ██     ██
#
######################################################################################
#
# Description: Path planner for mobile robot chassis navigation
# Author: Chenyu Cao, Herman Ye@Galbot
#
######################################################################################

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
from ioai_env import IOAIEnv


class BasePathPlanner(ABC):
    """Abstract base class for path planning algorithms.

    This class defines the interface that all path planning implementations
    must follow. It provides a common interface for different path planning
    strategies while allowing for flexible input parameters and return formats.
    """

    def __init__(self, environment: IOAIEnv):
        """Initialize the path planner with a reference to the IOAI environment.

        Args:
            environment(IOAIEnv): The IOAI simulation environment instance.
        """
        self.environment = environment

    @abstractmethod
    def plan_path(self, *args, **kwargs):
        """Plan the optimal path for the robot to navigate to the goal.

        This method should be implemented by subclasses to plan navigation paths.
        The input parameters and return values are flexible to accommodate different
        implementation approaches.

        Args:
            *args: Variable length argument list for flexible input parameters.
            **kwargs: Arbitrary keyword arguments for flexible input parameters.

        Returns:
            The planned path in any format suitable for the implementation.
            Common formats include:
            - List[Tuple[float, float]]: List of waypoints [(x1, y1), (x2, y2), ...]
            - np.ndarray: Array of waypoints [[x1, y1], [x2, y2], ...]
            - Dict: Dictionary containing path information
            - Any other format that suits the implementation

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("Subclasses must implement plan_path().")


class InterpolationPathPlanner(BasePathPlanner):
    """Simple interpolation path planner that directly connects two points.
    
    This path planner ignores obstacles and creates a direct path between
    start and goal positions using linear interpolation.
    """

    def __init__(self, environment: IOAIEnv):
        """Initialize the interpolation path planner.
        
        Args:
            environment: The IOAI simulation environment instance.
        """
        super().__init__(environment)

    def plan_path(
        self,
        start_position: Tuple[float, float],
        goal_position: Tuple[float, float],
        num_points: int = 50,
    ) -> List[Tuple[float, float]]:
        """Plan a direct path from start to goal position using linear interpolation.
        
        Args:
            start_position: Starting position coordinates (x, y) in meters.
            goal_position: Goal position coordinates (x, y) in meters.
            num_points: Number of points to generate along the path.
                
        Returns:
            List of waypoints forming the interpolated path.
        """
        if num_points < 2:
            num_points = 2
            
        return self._linear_interpolation(start_position, goal_position, num_points)

    def _linear_interpolation(
        self, 
        start: Tuple[float, float], 
        goal: Tuple[float, float], 
        num_points: int
    ) -> List[Tuple[float, float]]:
        """Perform linear interpolation between start and goal points.
        
        Args:
            start: Starting position (x, y).
            goal: Goal position (x, y).
            num_points: Number of points to generate.
            
        Returns:
            List of interpolated waypoints.
        """
        path = []
        
        for i in range(num_points):
            t = i / (num_points - 1)  # Parameter from 0 to 1
            
            # Linear interpolation
            x = start[0] + t * (goal[0] - start[0])
            y = start[1] + t * (goal[1] - start[1])
            
            path.append((x, y))
            
        return path
