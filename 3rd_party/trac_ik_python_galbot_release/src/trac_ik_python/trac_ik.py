#!/usr/bin/env python

# Author: Sammy Pfeiffer <Sammy.Pfeiffer at student.uts.edu.au>
# Convenience code to wrap TRAC IK

from trac_ik_python.trac_ik_wrap import TRAC_IK


class IK(object):
    def __init__(self, base_link, tip_link, urdf_file, 
                 max_steps=10000, timeout=0.005, epsilon=1e-4, rand_restart=True, wrap_around=True, relative_diff=True, solve_type="Speed"):
        self.base_link = base_link
        self.tip_link = tip_link
        self.urdf_file = urdf_file
        self._max_steps = max_steps
        self._timeout = timeout
        self._epsilon = epsilon
        self._rand_restart = rand_restart
        self._wrap_around = wrap_around
        self._relative_diff = relative_diff
        self._solve_type = solve_type
        self._ik_solver = TRAC_IK(self.base_link,
                                  self.tip_link,
                                  self.urdf_file,
                                  self._max_steps,
                                  self._timeout,
                                  self._epsilon,
                                  self._rand_restart,
                                  self._relative_diff,
                                  self._wrap_around,
                                  self._solve_type)
        self.number_of_joints = self._ik_solver.getNrOfJointsInChain()
        self.joint_names = self._ik_solver.getJointNamesInChain(
            self.urdf_file)
        self.link_names = self._ik_solver.getLinkNamesInChain()

    def get_ik(self, qinit,
               x, y, z,
               rx, ry, rz, rw,
               bx=1e-5, by=1e-5, bz=1e-5,
               brx=1e-3, bry=1e-3, brz=1e-3):
        """
        Do the IK call.

        :param list of float qinit: Initial status of the joints as seed.
        :param float x: X coordinates in base_frame.
        :param float y: Y coordinates in base_frame.
        :param float z: Z coordinates in base_frame.
        :param float rx: X quaternion coordinate.
        :param float ry: Y quaternion coordinate.
        :param float rz: Z quaternion coordinate.
        :param float rw: W quaternion coordinate.
        :param float bx: X allowed bound.
        :param float by: Y allowed bound.
        :param float bz: Z allowed bound.
        :param float brx: rotation over X allowed bound.
        :param float bry: rotation over Y allowed bound.
        :param float brz: rotation over Z allowed bound.

        :return: joint values or None if no solution found.
        :rtype: tuple of float.
        """
        if len(qinit) != self.number_of_joints:
            raise Exception("qinit has length %i and it should have length %i" % (
                len(qinit), self.number_of_joints))
        solution = self._ik_solver.CartToJnt(qinit,
                                             x, y, z,
                                             rx, ry, rz, rw,
                                             bx, by, bz,
                                             brx, bry, brz)
        if solution:
            return solution
        else:
            return None

    def get_joint_limits(self):
        """
        Return lower bound limits and upper bound limits for all the joints
        in the order of the joint names.
        """
        lb = self._ik_solver.getLowerBoundLimits()
        ub = self._ik_solver.getUpperBoundLimits()
        return lb, ub

    def set_joint_limits(self, lower_bounds, upper_bounds):
        """
        Set joint limits for all the joints.

        :arg list lower_bounds: List of float of the lower bound limits for
            all joints.
        :arg list upper_bounds: List of float of the upper bound limits for
            all joints.
        """
        if len(lower_bounds) != self.number_of_joints:
            raise Exception("lower_bounds array size mismatch, it's size %i, should be %i" % (
                len(lower_bounds),
                self.number_of_joints))

        if len(upper_bounds) != self.number_of_joints:
            raise Exception("upper_bounds array size mismatch, it's size %i, should be %i" % (
                len(upper_bounds),
                self.number_of_joints))
        self._ik_solver.setKDLLimits(lower_bounds, upper_bounds)

    def get_jacobian(self, q):
        return self._ik_solver.getJacobian(q)

    def get_forward_kinematics(self, q):
        return self._ik_solver.getForwardKinematics(q)
