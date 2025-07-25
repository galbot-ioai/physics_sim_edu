#!/usr/bin/env python

from trac_ik_python.trac_ik import IK
from numpy.random import random
import time
import sys
import os

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_file = os.path.join(current_dir, "pr2.urdf")
    if len(sys.argv) == 2:
        urdf_file = sys.argv[1]

    ik_solver = IK("right_arm_base_link", "right_arm_end_effector_mount_link", urdf_file)

    print("IK solver uses link chain:")
    print(ik_solver.link_names)

    print("IK solver base frame:")
    print(ik_solver.base_link)

    print("IK solver tip link:")
    print(ik_solver.tip_link)

    print("IK solver for joints:")
    print(ik_solver.joint_names)

    print("IK solver using joint limits:")
    lb, up = ik_solver.get_joint_limits()
    print("Lower bound: " + str(lb))
    print("Upper bound: " + str(up))

    qinit = [0.] * ik_solver.number_of_joints
    x = y = z = 0.0
    rx = ry = rz = 0.0
    rw = 1.0
    bx = by = bz = 0.001
    brx = bry = brz = 0.1

    # Generate a set of random coords in the arm workarea approx
    NUM_COORDS = 200
    rand_coords = []
    for _ in range(NUM_COORDS):
        x = random() * 1.0 - 0.5
        y = random() * 1.0 - 0.5
        z = random() * 1.0 - 0.5
        rand_coords.append((x, y, z))
    print("Initial joint state:", qinit)

    # Check some random coords with fixed orientation
    avg_time = 0.0
    num_solutions_found = 0
    for x, y, z in rand_coords:
        ini_t = time.time()
        sol = ik_solver.get_ik(qinit,
                               x, y, z,
                               rx, ry, rz, rw,
                               bx, by, bz,
                               brx, bry, brz)
        fin_t = time.time()
        call_time = fin_t - ini_t
        # print("IK call took: " + str(call_time))
        # print(ik_solver)
        avg_time += call_time
        if sol:
            # print("X, Y, Z: " + str( (x, y, z) ))
            # print("SOL: " + str(sol))
            num_solutions_found += 1
    avg_time = avg_time / NUM_COORDS

    print()
    print("Found " + str(num_solutions_found) + " of 200 random coords")
    print("Average IK call time: " + str(avg_time))
    print()

    # Check if orientation bounds work
    avg_time = 0.0
    num_solutions_found = 0
    brx = bry = brz = 9999.0  # We don't care about orientation
    for x, y, z in rand_coords:
        ini_t = time.time()
        sol = ik_solver.get_ik(qinit,
                               x, y, z,
                               rx, ry, rz, rw,
                               bx, by, bz,
                               brx, bry, brz)
        fin_t = time.time()
        call_time = fin_t - ini_t
        # print("IK call took: " + str(call_time))
        avg_time += call_time
        if sol:
            # print("X, Y, Z: " + str( (x, y, z) ))
            # print("SOL: " + str(sol))
            num_solutions_found += 1

    avg_time = avg_time / NUM_COORDS
    print()
    print("Found " + str(num_solutions_found) + " of 200 random coords ignoring orientation")
    print("Average IK call time: " + str(avg_time))
    print()

    # Check if big position and orientation bounds work
    avg_time = 0.0
    num_solutions_found = 0
    bx = by = bz = 9999.0
    brx = bry = brz = 9999.0
    for x, y, z in rand_coords:
        ini_t = time.time()
        sol = ik_solver.get_ik(qinit,
                               x, y, z,
                               rx, ry, rz, rw,
                               bx, by, bz,
                               brx, bry, brz)
        fin_t = time.time()
        call_time = fin_t - ini_t
        # print("IK call took: " + str(call_time))
        avg_time += call_time
        if sol:
            # print("X, Y, Z: " + str( (x, y, z) ))
            # print("SOL: " + str(sol))
            num_solutions_found += 1

    avg_time = avg_time / NUM_COORDS

    print()
    print("Found " + str(num_solutions_found) + " of 200 random coords ignoring everything")
    print("Average IK call time: " + str(avg_time))
    print()

    qq = [0,0,0,0,0,0,0]
    print(f"jacobian: {ik_solver.get_jacobian(qq)}")
    print(f"forward_kinematics: {ik_solver.get_forward_kinematics(qq)}")
