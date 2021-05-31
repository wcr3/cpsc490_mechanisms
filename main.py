from typing import Generator
import numpy as np

from mech_maker.generics import Vec3, Quaternion
from mech_maker.curve import Curve, CurvePoint

from mech_maker.shape import Line, Shape
from mech_maker.gcs.outputs import TrackPoint
from mech_maker.gcs.inputs import MechanismInputParams, FixedMechanismInput, RelativeMechanismInput
from mech_maker.gcs.mechanism import Mechanism, Mechanism2D
from mech_maker.gcs.member import Member
from mech_maker.gcs.solver import Solver, ScipySLSQPSolver
from mech_maker.gcs.constraint import FixedAllConstraint, FixedLocationConstraint, FixedPinConstraint, RelativeAxisAlignedConstraint, RelativeLocationConstraint, RelativeOrientationConstraint, RelativePinConstraint

from mech_maker.gui.animator import MPEGAnimator

from mech_maker.analyzer.features import CurveFeature

def square_mech(solver: Solver, steps: int) -> Mechanism2D:
    mech = Mechanism2D(solver, z=0)

    member1 = Member(Vec3(0,0,0), Quaternion.from_axis_angle(Vec3(0,0,1), 0), Line(1))
    member2 = Member(Vec3(0,0,0), Quaternion.from_axis_angle(Vec3(0,0,1), np.pi / 2), Line(1))
    member3 = Member(Vec3(0,1,0), Quaternion.from_axis_angle(Vec3(0,0,1), 0), Line(1))
    member4 = Member(Vec3(1.5,0,0), Quaternion.from_axis_angle(Vec3(0,0,1), np.pi / 2), Line(1))
    mech.add_member(member1)
    mech.add_member(member2)
    mech.add_member(member3)
    mech.add_member(member4)
    mech.add_constraint(FixedAllConstraint(member1, (Vec3(0,0,0), Vec3(0,0,0)), (Quaternion.from_axis_angle(Vec3(0,0,1), 0), Quaternion.from_axis_angle(Vec3(0,0,1), 0))))
    loc_params = MechanismInputParams((Vec3(0,0,0), Vec3(0,0,0)), (Vec3(0,0,0), Vec3(0,0,0)))
    end_angle = np.pi * 2.0 * (1.0 - 1.0/float(steps))
    orient_params = MechanismInputParams((Quaternion.from_axis_angle(Vec3(0,0,1), 0), Quaternion.from_axis_angle(Vec3(0,0,1), 0)), (Quaternion.from_axis_angle(Vec3(0,0,1), 0), Quaternion.from_axis_angle(Vec3(0,0,1), end_angle)))
    inp = FixedMechanismInput(FixedAllConstraint, member2, loc_params, orient_params)
    mech.add_input(inp)
    mech.add_constraint(RelativePinConstraint(member2, member3, (Vec3(1,0,0), Vec3(0,0,0)), (Vec3(0,0,1), Vec3(0,0,1))))
    mech.add_constraint(RelativePinConstraint(member1, member4, (Vec3(1,0,0), Vec3(0,0,0)), (Vec3(0,0,1), Vec3(0,0,1))))
    mech.add_constraint(RelativePinConstraint(member3, member4, (Vec3(1,0,0), Vec3(1,0,0)), (Vec3(0,0,1), Vec3(0,0,1))))
    mech.add_track_point(TrackPoint(member4, Vec3(1,0,0)))

    return mech

def rotating_square(solver: Solver, steps: int) -> Mechanism:
    mech = Mechanism(solver)

    square = Shape([Vec3(0,0,0), Vec3(1,0,0), Vec3(1,1,0), Vec3(0,1,0), Vec3(0,0,0)])
    member = Member(Vec3(0,0,0), Quaternion.identity(), square)
    mech.add_member(member)
    loc_params = MechanismInputParams((Vec3(0,0,0), Vec3(0,0,0)), (Vec3(0,0,0), Vec3(0,0,0)))
    end_ratio = 1.0 - 1.0/float(steps)
    end_angle = np.pi * 2.0 * end_ratio
    orient_params = MechanismInputParams((Quaternion.identity(), Quaternion.identity()), (Quaternion.identity(), Quaternion.from_axis_angle(Vec3(1,0,0), end_angle)))
    mech.add_input(FixedMechanismInput(FixedAllConstraint, member, (0.0, end_ratio), loc_params, orient_params))
    new_orient_params = MechanismInputParams((Quaternion.identity(), Quaternion.from_axis_angle(Vec3(1,0,0), end_angle)), (Quaternion.identity(), Quaternion.from_axis_angle(Vec3(1,0,0), np.pi * 2.0)))
    mech.add_input(FixedMechanismInput(FixedAllConstraint, member, (end_ratio, 1.0), loc_params, new_orient_params))

    return mech
    
def six_bar(solver: Solver, steps: int) -> Mechanism2D:
    mech = Mechanism2D(solver, z=0)
    
    # link AB
    ab = Member(Vec3(0,0,0), Quaternion.identity(), Line(0.25))
    mech.add_member(ab)
    loc_params = MechanismInputParams((Vec3(0,0,0), Vec3(0,0,0)), (Vec3(0,0,0), Vec3(0,0,0)))
    end_ratio = 1.0 - 1.0/float(steps)
    end_angle = np.pi * 2.0 * end_ratio
    orient_params = MechanismInputParams((Quaternion.identity(), Quaternion.identity()), (Quaternion.identity(), Quaternion.from_axis_angle(Vec3(0,0,1), end_angle)))
    mech.add_input(FixedMechanismInput(FixedAllConstraint, ab, (0.0, end_ratio), loc_params, orient_params))
    new_orient_params = MechanismInputParams((Quaternion.identity(), Quaternion.from_axis_angle(Vec3(0,0,1), end_angle)), (Quaternion.identity(), Quaternion.from_axis_angle(Vec3(0,0,1), np.pi * 2.0)))
    mech.add_input(FixedMechanismInput(FixedAllConstraint, ab, (end_ratio, 1.0), loc_params, new_orient_params))

    # link CD
    cd = Member(Vec3(0,0,0), Quaternion.identity(), Line(2.25))
    mech.add_member(cd)
    mech.add_constraint(FixedPinConstraint(cd, (Vec3(2.25,0,0), Vec3(2.75,-3.25,0)), (Vec3(0,0,1), Vec3(0,0,1))))

    # link BCE
    bce_shape = Shape([Vec3(0,0,0), Vec3(0,-3,0), Vec3(-0.520944533001,-2.95442325904,0), Vec3(0,0,0)])
    bce = Member(Vec3(0,0,0), Quaternion.identity(), bce_shape)
    mech.add_member(bce)
    mech.add_constraint(RelativePinConstraint(ab, bce, (Vec3(0.25,0,0), Vec3(0,0,0)), (Vec3(0,0,1), Vec3(0,0,1))))
    mech.add_constraint(RelativePinConstraint(bce, cd, (Vec3(0,-3,0), Vec3(0,0,0)), (Vec3(0,0,1), Vec3(0,0,1))))

    # link GFH
    gfh_shape = Shape([Vec3(0,0,0), Vec3(0,-6.5,0), Vec3(-1.55291427062,-5.79555495773,0), Vec3(0,0,0)])
    gfh = Member(Vec3(0,0,0), Quaternion.from_axis_angle(Vec3(0,0,1), -np.pi / 2), gfh_shape)
    mech.add_member(gfh)
    mech.add_constraint(FixedPinConstraint(gfh, (Vec3(0,0,0), Vec3(-1.75,3,0)), (Vec3(0,0,1), Vec3(0,0,1))))

    # link EF
    ef = Member(Vec3(0,0,0), Quaternion.identity(), Line(2.25))
    mech.add_member(ef)
    mech.add_constraint(RelativePinConstraint(gfh, ef, (Vec3(0,-6.5,0), Vec3(0,0,0)), (Vec3(0,0,1), Vec3(0,0,1))))
    mech.add_constraint(RelativePinConstraint(bce, ef, (Vec3(-0.520944533001,-2.95442325904,0), Vec3(2.25,0,0)), (Vec3(0,0,1), Vec3(0,0,1))))

    mech.add_track_point(TrackPoint(gfh, Vec3(-1.55291427062,-5.79555495773,0)))

    return mech

def crank_rocker(solver: Solver, steps: int) -> tuple[Mechanism2D, Member]:
    mech = Mechanism2D(solver, z=0)

    # crank
    crank = Member(Vec3(-2,3,0), Quaternion.identity(), Line(1))
    mech.add_member(crank)
    loc_params = MechanismInputParams((Vec3(0,0,0), Vec3(-3,3,0)), (Vec3(0,0,0), Vec3(-3,3,0)))
    orient_params = MechanismInputParams((Quaternion.identity(), Quaternion.identity()), (Quaternion.identity(), Quaternion.from_axis_angle(Vec3(0,0,1), np.pi * 2.0)))
    mech.add_input(FixedMechanismInput(FixedAllConstraint, crank, (0.0, 1.0), loc_params, orient_params))

    # rocker
    rocker = Member(Vec3(2,-3,0), Quaternion.from_axis_angle(Vec3(0,0,1),np.pi / 2), Line(5.75))
    mech.add_member(rocker)
    mech.add_constraint(FixedPinConstraint(rocker, (Vec3(0,0,0), Vec3(2,-3,0)), (Vec3(0,0,1), Vec3(0,0,1))))

    # coupler
    coupler_shape = Shape([Vec3(0,0,0), Vec3(5,0,0), Vec3(4.5,1,0), Vec3(2.5,2.5,0), Vec3(0.5,1,0), Vec3(0,0,0)])
    coupler = Member(Vec3(-1,3,0), Quaternion.identity(), coupler_shape)
    mech.add_member(coupler)
    mech.add_constraint(RelativePinConstraint(crank, coupler, (Vec3(1,0,0), Vec3(0,0,0)), (Vec3(0,0,1), Vec3(0,0,1))))
    mech.add_constraint(RelativePinConstraint(rocker, coupler, (Vec3(5.75,0,0), Vec3(5,0,0)), (Vec3(0,0,1), Vec3(0,0,1))))

    mech.add_track_point(TrackPoint(coupler, Vec3(2.5,2.5,0)))

    return mech, coupler

def crank_rocker_rotated(solver: Solver, steps: int) -> tuple[Mechanism, Member]:
    mech = Mechanism(solver)

    # crank
    crank = Member(Vec3(-3,3,0), Quaternion.identity(), Line(1))
    mech.add_member(crank)
    loc_params = MechanismInputParams((Vec3(0,0,0), Vec3(-3,3,0)), (Vec3(0,0,0), Vec3(-3,3,0)))
    orient_params = MechanismInputParams((Quaternion.identity(), Quaternion.identity()), (Quaternion.identity(), Quaternion.from_axis_angle(Vec3(0,0,1), np.pi * 2.0)))
    mech.add_input(FixedMechanismInput(FixedAllConstraint, crank, (0.0, 1.0), loc_params, orient_params))

    # # rocker
    rocker = Member(Vec3(2,-3,0), Quaternion.from_axis_angle(Vec3(0,0,1),np.pi / 2), Line(5.75))
    mech.add_member(rocker)
    mech.add_constraint(FixedLocationConstraint(rocker, (Vec3(0,0,0), Vec3(2,-3,0))))

    # coupler
    coupler_shape = Shape([Vec3(0,0,0), Vec3(5,0,0), Vec3(4.5,1,0), Vec3(2.5,2.5,0), Vec3(0.5,1,0), Vec3(0,0,0)])
    empty_coupler = Member(Vec3(-1,3,0), Quaternion.identity(), Line(5))
    coupler = Member(Vec3(-1,3,0), Quaternion.identity(), coupler_shape)
    mech.add_member(empty_coupler)
    mech.add_member(coupler)
    mech.add_constraint(RelativePinConstraint(crank, empty_coupler, (Vec3(1,0,0), Vec3(0,0,0)), (Vec3(0,0,1), Vec3(0,0,1))))
    mech.add_constraint(RelativeLocationConstraint(rocker, empty_coupler, (Vec3(5.75,0,0), Vec3(5,0,0))))
    mech.add_constraint(RelativeLocationConstraint(coupler, empty_coupler, (Vec3(0,0,0), Vec3(0,0,0))))
    mech.add_constraint(RelativeLocationConstraint(coupler, empty_coupler, (Vec3(5,0,0), Vec3(5,0,0))))

    orient_params_2 = MechanismInputParams((Quaternion.identity(), Quaternion.identity()), (Quaternion.identity(), Quaternion.from_axis_angle(Vec3(1,0,0), np.pi * 2.0)))
    mech.add_input(RelativeMechanismInput(RelativeOrientationConstraint, empty_coupler, coupler, (0,1.0), orient_params_2))

    mech.add_track_point(TrackPoint(coupler, Vec3(2.5,2.5,0)))

    return mech, coupler

def main() -> None:
    np.seterr('raise')

    steps = 20

    # convergence_animator = MPEGAnimator(Vec3(1,1,1), '.\\Images\\test_converge.mp4', 5, (-7,7), (-7,7))
    solver = ScipySLSQPSolver() #lambda shapes: convergence_animator.write_frame(shapes, []))

    # mech = square_mech(solver, steps)
    # mech = six_bar(solver, steps)
    # mech, member = crank_rocker(solver, steps)
    mech, member = crank_rocker_rotated(solver, steps)
    # mech = rotating_square(solver, steps)

    animator1 = MPEGAnimator(Vec3(0,0,1), '.\\Images\\crank_rocker_rotating_coupler\\xy.mp4', 5, (-7,7), (-7,7))
    animator2 = MPEGAnimator(Vec3(-1,1,1), '.\\Images\\crank_rocker_rotating_coupler\\iso_neg.mp4', 5, (-7,7), (-7,7))
    animator3 = MPEGAnimator(Vec3(1,1,1), '.\\Images\\crank_rocker_rotating_coupler\\iso_pos.mp4', 5, (-7,7), (-7,7))

    # convergence_animator.write_frame(mech.shapes(), [])
    def step_callback(solved: bool, shapes: Generator[Shape, None, None], curves: Generator[Curve, None, None]) -> None:
        if solved:
            shapes = list(shapes)
            curves = list(curves)
            animator1.write_frame(shapes, curves)
            animator2.write_frame(shapes, curves)
            animator3.write_frame(shapes, curves)

    mech.add_track_point(TrackPoint(member, Vec3(4.5,1,0)))
    mech.add_track_point(TrackPoint(member, Vec3(0.5,1,0)))

    mech.solve_times([val / (steps - 1) for val in range(steps)], step_callback)

    curves = list(mech.curves())
    c_fs: list[CurveFeature] = []

    for curve in curves:
        c_fs.append(CurveFeature(curve, 15))

    l: list[Curve] = []
    for c_f in c_fs:
        l.extend(c_f.curves())

    # animator1.write_frame([], l)
    # animator1.write_frame([], [c_f._curve_rotated for c_f in c_fs])
    # animator1.write_frame([], [c_f._curve_scaled for c_f in c_fs])

    for i, c_f in enumerate(c_fs):
        print('curve:')
        print(i)
        print(c_f.features)
        for o_c_f in c_fs[i+1:]:
            print(c_f.compare(o_c_f))

    animator1.finish()
    animator2.finish()
    animator3.finish()
    # convergence_animator.finish()

if __name__ == "__main__":
    main()