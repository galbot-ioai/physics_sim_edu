"""
Control utilities for physics simulation including PID controllers and path following.
"""

import math
import numpy as np


class PIDController:
    """Basic PID controller"""
    
    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki 
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        
    def update(self, error, dt=0.001):
        """Update PID controller with current error"""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output
        
    def reset(self):
        """Reset PID controller state"""
        self.prev_error = 0.0
        self.integral = 0.0


class BasicPathFollower:
    """Basic path following controller"""
    
    def __init__(self, velocity=1.0):
        self.velocity = velocity
        self.heading_pid = PIDController(kp=2.5, ki=0.1, kd=0.08)
        
    def calculate_control(self, current_pos, current_heading, target_pos):
        """Calculate basic velocity commands"""
        if target_pos is None:
            return 0.0, 0.0, 0.0
            
        # Calculate target heading
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        target_heading = math.atan2(dy, dx)
        
        # Normalize heading error to [-pi, pi]
        heading_error = target_heading - current_heading
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
            
        # PID control for heading
        yaw_velocity = self.heading_pid.update(heading_error)
        
        # Velocity in world frame (not robot frame)
        forward_velocity = self.velocity * math.cos(target_heading)
        side_velocity = self.velocity * math.sin(target_heading)
        
        return forward_velocity, side_velocity, yaw_velocity
