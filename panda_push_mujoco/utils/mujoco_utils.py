from typing import Dict, Tuple, Union
from mujoco import MjData, MjModel
import numpy as np
import os, panda_push_mujoco

from gymnasium import error
from gymnasium_robotics.utils.mujoco_utils import (get_site_jacp,
                                                   get_site_jacr,
                                                   set_joint_qpos,
                                                   set_joint_qvel,
                                                   get_joint_qpos,
                                                   get_joint_qvel,
                                                   get_site_xpos,
                                                   get_site_xmat,
                                                   MujocoModelNames)

try:
    import mujoco
    from mujoco import MjModel, MjData
except ImportError as e:
    raise error.DependencyNotInstalled(f"{e}. (HINT: you need to install mujoco")

from panda_push_mujoco.utils import controller_utils, rotations_utils


def get_model_robot_joints(model: MjModel, robot_name: str):
    mj_model_names = MujocoModelNames(model)

    joint_names = [name for name in mj_model_names.joint_names if robot_name in name]
    joint_ids = [mj_model_names.joint_name2id[joint_name] for joint_name in joint_names]
    joint_qposadr = model.jnt_qposadr[joint_ids]
    joint_dofadr = model.jnt_dofadr[joint_ids]

    return joint_names, joint_ids, joint_qposadr, joint_dofadr


def get_model_actuators(model: MjModel):
    mj_model_names = MujocoModelNames(model)

    actuator_names = [name for name in mj_model_names.actuator_names]
    actuator_ids = [mj_model_names.actuator_name2id[actuator_name] for actuator_name in actuator_names]

    return actuator_names, actuator_ids


def set_geom_material(model: MjModel, geom_name: str, material_name: str):
    material_id = model.material(material_name).id
    geom_id = model.geom(geom_name).id
    model.geom_matid[geom_id] = material_id


def get_panda_robot_xml_str(gravity_compensation: float=1.0,
                            path_to_meshes: str=os.path.join(panda_push_mujoco.__path__[0], "assets", "meshes"),
                            fingertip_model: bool=False,
                            use_red_ee: bool=False):
    
    path_visual_meshes = os.path.join(path_to_meshes, "visual")
    path_collision_meshes = os.path.join(path_to_meshes, "collision")

    if use_red_ee:
        ee_material = "red"
    else:
        ee_material = "black" if fingertip_model else "white"

    if fingertip_model:
        xml_collision_meshes = fr"""
            <mesh name="ubi_tip_collision"   file="{os.path.join(path_collision_meshes, 'ubi_tip_collision.stl')}"   scale="0.001 0.001 0.001" />
        """

        xml_visual_meshes = fr"""
            <mesh name="poking_stick_for_ubi_visual" file="{os.path.join(path_visual_meshes, 'poking_stick_for_ubi.stl')}" scale="0.001 0.001 0.001" />
            <mesh name="ubi_tip_visual" file="{os.path.join(path_visual_meshes, 'ubi_tip_visual.stl')}" scale="0.001 0.001 0.001" />
        """

        xml_ee = fr"""
                                            <body name="panda_poking_stick" pos="0 0 0.107" gravcomp="{gravity_compensation}">
                                                <inertial pos="0 0 0.0161" mass="0.505" diaginertia="0.000651 0.000651 0.000896"/>
                                                <geom name="panda_poking_stick_geom" class="visual" mesh="poking_stick_for_ubi_visual" material="{ee_material}" pos="0 0 0" euler="0 0 0"/>
                                                <geom class="collision" type="cylinder" size="0.06725 0.0075" pos="0 0 0.0075"/>
                                                <geom class="collision" type="cylinder" size="0.05 0.005" pos="0 0 0.02"/>
                                                <geom class="collision" type="cylinder" size="0.025 0.005" pos="0 0 0.03"/>
                                                <geom class="collision" type="cylinder" size="0.0075 0.07225" pos="0 0 0.07375"/>

                                                <site name="panda_ee_site" pos="0 0 0.0161" rgba="0 0 0 0" size="0.02 0.02 0.02"></site>
                                                
                                                <body name="panda_ffdistal" pos="0 0 0.13605" gravcomp="{gravity_compensation}">
                                                    <geom name="panda_fingertip_geom_visual" class="visual" mesh="ubi_tip_visual" pos="0 0 0" euler="0 0 0" material="white"/>
                                                    <geom name="panda_fingertip_geom" class="collision" mesh="ubi_tip_collision" pos="0 0 0" euler="0 0 0"/>
                                                    <site name="touch_site" type="ellipsoid" size="0.0107 0.008 0.0138"  pos="0 -0.00048 0.0185" euler="-0.17 0 0" rgba="0 1 0 0.2"/>
                                                </body>
                                            </body>
        """

        xml_exclude = r"""
            <exclude body1="panda_poking_stick" body2="panda_ffdistal" />
        """

        xml_touch_sensor = r"""
        <sensor>
            <touch name="tactile_sensor" site="touch_site"/>
        </sensor>
        """
    else:
        xml_collision_meshes = ""
        xml_visual_meshes = ""
        xml_ee = fr"""
                                            <body name="panda_quick_mount" pos="0 0 0.107" gravcomp="{gravity_compensation}">
                                                <inertial pos="0 0 0.0075" mass="0.25" diaginertia="0.000289 0.000289 0.000569"/>
                                                <geom name="panda_quick_mount_geom" class="visual" type="cylinder" size="0.0675 0.0075" material="{ee_material}" pos="0 0 0.0075" euler="0 0 0"/>
                                                <geom class="collision" type="cylinder" size="0.0675 0.0075" pos="0 0 0.0075" euler="0 0 0"/>

                                                <site name="panda_ee_site" pos="0 0 0" size="0.02 0.02 0.02" rgba="0 0 0 0"/>

                                                <body name="panda_poking_stick" pos="0 0 0.015" gravcomp="{gravity_compensation}">
                                                    <inertial pos="0 0 0.0775" mass="0.25" diaginertia="0.000504 0.000504 0.00000703"/>
                                                    <geom name="panda_poking_stick_geom" class="visual" type="cylinder" size="0.0075 0.0775" material="{ee_material}" pos="0 0 0.0775" euler="0 0 0"/>
                                                    <geom class="collision" type="cylinder" size="0.0075 0.0775" pos="0 0 0.0775" euler="0 0 0"/>
                                                </body>
                                            </body>
        """
        xml_exclude = ""
        xml_touch_sensor = ""

    xml_robot = fr"""
        <!-- panda robot -->
        <asset>
            <material class="panda" name="white"      rgba="1 1 1 1" />
            <material class="panda" name="off_white"  rgba="0.901961 0.921569 0.929412 1" />
            <material class="panda" name="black"      rgba="0.25 0.25 0.25 1" />
            <material class="panda" name="green"      rgba="0 1 0 1" />
            <material class="panda" name="red"        rgba="1 0 0 1" />
            <material class="panda" name="light_blue" rgba="0.39216 0.541176 0.780392 1" />

            <mesh name="link0_collision"  file="{os.path.join(path_collision_meshes, 'link0.stl')}"  scale="1 1 1" />
            <mesh name="link1_collision"  file="{os.path.join(path_collision_meshes, 'link1.stl')}"  scale="1 1 1" />
            <mesh name="link2_collision"  file="{os.path.join(path_collision_meshes, 'link2.stl')}"  scale="1 1 1" />
            <mesh name="link3_collision"  file="{os.path.join(path_collision_meshes, 'link3.stl')}"  scale="1 1 1" />
            <mesh name="link4_collision"  file="{os.path.join(path_collision_meshes, 'link4.stl')}"  scale="1 1 1" />
            <mesh name="link5_0_collision"  file="{os.path.join(path_collision_meshes, 'link5_collision_0.obj')}"  scale="1 1 1" />
            <mesh name="link5_1_collision"  file="{os.path.join(path_collision_meshes, 'link5_collision_1.obj')}"  scale="1 1 1" />
            <mesh name="link5_2_collision"  file="{os.path.join(path_collision_meshes, 'link5_collision_2.obj')}"  scale="1 1 1" />
            <mesh name="link6_collision"  file="{os.path.join(path_collision_meshes, 'link6.stl')}"  scale="1 1 1" />
            <mesh name="link7_collision"  file="{os.path.join(path_collision_meshes, 'link7.stl')}"  scale="1 1 1" />
            {xml_collision_meshes}

            <mesh name="link0_0_visual"  file="{os.path.join(path_visual_meshes, 'link0_0.obj')}"  scale="1 1 1" />
            <mesh name="link0_1_visual"  file="{os.path.join(path_visual_meshes, 'link0_1.obj')}"  scale="1 1 1" />
            <mesh name="link0_2_visual"  file="{os.path.join(path_visual_meshes, 'link0_2.obj')}"  scale="1 1 1" />
            <mesh name="link0_3_visual"  file="{os.path.join(path_visual_meshes, 'link0_3.obj')}"  scale="1 1 1" />
            <mesh name="link0_4_visual"  file="{os.path.join(path_visual_meshes, 'link0_4.obj')}"  scale="1 1 1" />
            <mesh name="link0_5_visual"  file="{os.path.join(path_visual_meshes, 'link0_5.obj')}"  scale="1 1 1" />
            <mesh name="link0_7_visual"  file="{os.path.join(path_visual_meshes, 'link0_7.obj')}"  scale="1 1 1" />
            <mesh name="link0_8_visual"  file="{os.path.join(path_visual_meshes, 'link0_8.obj')}"  scale="1 1 1" />
            <mesh name="link0_9_visual"  file="{os.path.join(path_visual_meshes, 'link0_9.obj')}"  scale="1 1 1" />
            <mesh name="link0_10_visual" file="{os.path.join(path_visual_meshes, 'link0_10.obj')}"  scale="1 1 1" />
            <mesh name="link0_11_visual" file="{os.path.join(path_visual_meshes, 'link0_11.obj')}"  scale="1 1 1" />
            <mesh name="link1_visual"  file="{os.path.join(path_visual_meshes, 'link1.obj')}"  scale="1 1 1" />
            <mesh name="link2_visual"  file="{os.path.join(path_visual_meshes, 'link2.obj')}"  scale="1 1 1" />
            <mesh name="link3_0_visual"  file="{os.path.join(path_visual_meshes, 'link3_0.obj')}"  scale="1 1 1" />
            <mesh name="link3_1_visual"  file="{os.path.join(path_visual_meshes, 'link3_1.obj')}"  scale="1 1 1" />
            <mesh name="link3_2_visual"  file="{os.path.join(path_visual_meshes, 'link3_2.obj')}"  scale="1 1 1" />
            <mesh name="link3_3_visual"  file="{os.path.join(path_visual_meshes, 'link3_3.obj')}"  scale="1 1 1" />
            <mesh name="link4_0_visual"  file="{os.path.join(path_visual_meshes, 'link4_0.obj')}"  scale="1 1 1" />
            <mesh name="link4_1_visual"  file="{os.path.join(path_visual_meshes, 'link4_1.obj')}"  scale="1 1 1" />
            <mesh name="link4_2_visual"  file="{os.path.join(path_visual_meshes, 'link4_2.obj')}"  scale="1 1 1" />
            <mesh name="link4_3_visual"  file="{os.path.join(path_visual_meshes, 'link4_3.obj')}"  scale="1 1 1" />
            <mesh name="link5_0_visual"  file="{os.path.join(path_visual_meshes, 'link5_0.obj')}"  scale="1 1 1" />
            <mesh name="link5_1_visual"  file="{os.path.join(path_visual_meshes, 'link5_1.obj')}"  scale="1 1 1" />
            <mesh name="link5_2_visual"  file="{os.path.join(path_visual_meshes, 'link5_2.obj')}"  scale="1 1 1" />
            <mesh name="link6_0_visual"  file="{os.path.join(path_visual_meshes, 'link6_0.obj')}"  scale="1 1 1" />
            <mesh name="link6_1_visual"  file="{os.path.join(path_visual_meshes, 'link6_1.obj')}"  scale="1 1 1" />
            <mesh name="link6_2_visual"  file="{os.path.join(path_visual_meshes, 'link6_2.obj')}"  scale="1 1 1" />
            <mesh name="link6_3_visual"  file="{os.path.join(path_visual_meshes, 'link6_3.obj')}"  scale="1 1 1" />
            <mesh name="link6_4_visual"  file="{os.path.join(path_visual_meshes, 'link6_4.obj')}"  scale="1 1 1" />
            <mesh name="link6_5_visual"  file="{os.path.join(path_visual_meshes, 'link6_5.obj')}"  scale="1 1 1" />
            <mesh name="link6_6_visual"  file="{os.path.join(path_visual_meshes, 'link6_6.obj')}"  scale="1 1 1" />
            <mesh name="link6_7_visual"  file="{os.path.join(path_visual_meshes, 'link6_7.obj')}"  scale="1 1 1" />
            <mesh name="link6_8_visual"  file="{os.path.join(path_visual_meshes, 'link6_8.obj')}"  scale="1 1 1" />
            <mesh name="link6_9_visual"  file="{os.path.join(path_visual_meshes, 'link6_9.obj')}"  scale="1 1 1" />
            <mesh name="link6_10_visual"  file="{os.path.join(path_visual_meshes, 'link6_10.obj')}"  scale="1 1 1" />
            <mesh name="link6_11_visual"  file="{os.path.join(path_visual_meshes, 'link6_10.obj')}"  scale="1 1 1" />
            <mesh name="link6_12_visual"  file="{os.path.join(path_visual_meshes, 'link6_12.obj')}"  scale="1 1 1" />
            <mesh name="link6_13_visual"  file="{os.path.join(path_visual_meshes, 'link6_13.obj')}"  scale="1 1 1" />
            <mesh name="link6_14_visual"  file="{os.path.join(path_visual_meshes, 'link6_14.obj')}"  scale="1 1 1" />
            <mesh name="link6_15_visual"  file="{os.path.join(path_visual_meshes, 'link6_15.obj')}"  scale="1 1 1" />
            <mesh name="link6_16_visual"  file="{os.path.join(path_visual_meshes, 'link6_16.obj')}"  scale="1 1 1" />
            <mesh name="link7_0_visual"  file="{os.path.join(path_visual_meshes, 'link7_0.obj')}"  scale="1 1 1" />
            <mesh name="link7_1_visual"  file="{os.path.join(path_visual_meshes, 'link7_1.obj')}"  scale="1 1 1" />
            <mesh name="link7_2_visual"  file="{os.path.join(path_visual_meshes, 'link7_2.obj')}"  scale="1 1 1" />
            <mesh name="link7_3_visual"  file="{os.path.join(path_visual_meshes, 'link7_3.obj')}"  scale="1 1 1" />
            <mesh name="link7_4_visual"  file="{os.path.join(path_visual_meshes, 'link7_4.obj')}"  scale="1 1 1" />
            <mesh name="link7_5_visual"  file="{os.path.join(path_visual_meshes, 'link7_5.obj')}"  scale="1 1 1" />
            <mesh name="link7_6_visual"  file="{os.path.join(path_visual_meshes, 'link7_6.obj')}"  scale="1 1 1" />
            <mesh name="link7_7_visual"  file="{os.path.join(path_visual_meshes, 'link7_7.obj')}"  scale="1 1 1" />
            {xml_visual_meshes}
        </asset>

        <default>
            <default class="panda">
                <material specular="0.5" shininess="0.25"/>
                <joint pos="0 0 0" axis="0 0 1" limited="true" damping="0.003" />

                <default class="visual">
                    <geom type="mesh" contype="0" conaffinity="0" group="0" mass="0" />
                </default>
                <default class="collision">
                    <geom contype="1" conaffinity="1" group="3" type="mesh" />
                </default>
            </default>
        </default>

        <worldbody>
            <light pos="0 0 1000" castshadow="false" />

            <body name="panda_link0" childclass="panda" pos="0 0 0" gravcomp="{gravity_compensation}">
                <inertial pos="-0.041018 -0.00014 0.049974" mass="0.629769" fullinertia="0.00315 0.00388 0.004285 8.2904E-07 0.00015 8.2299E-06" />
                <geom class="visual"    mesh="link0_0_visual"   material="off_white"/>
                <geom class="visual"    mesh="link0_1_visual"   material="black"/>
                <geom class="visual"    mesh="link0_2_visual"   material="off_white"/>
                <geom class="visual"    mesh="link0_3_visual"   material="black"/>
                <geom class="visual"    mesh="link0_4_visual"   material="off_white"/>
                <geom class="visual"    mesh="link0_5_visual"   material="black"/>
                <geom class="visual"    mesh="link0_7_visual"   material="white"/>
                <geom class="visual"    mesh="link0_8_visual"   material="white"/>
                <geom class="visual"    mesh="link0_9_visual"   material="black"/>
                <geom class="visual"    mesh="link0_10_visual"  material="off_white"/>
                <geom class="visual"    mesh="link0_11_visual"  material="white"/>
                <geom class="collision" mesh="link0_collision" />

                <body name="panda_link1" pos="0 0 0.333" gravcomp="{gravity_compensation}">
                    <inertial pos="3.875e-03 2.081e-03 -4.762e-02" mass="4.970684" fullinertia="7.0337e-01 7.0661e-01 9.1170e-03 -1.3900e-04 6.7720e-03 1.9169e-02" />
                    <joint name="panda_joint1" range="-2.8973 2.8973" />
                    <geom class="visual"    mesh="link1_visual" material="white"/>
                    <geom class="collision" mesh="link1_collision" />

                    <body name="panda_link2" pos="0 0 0" quat="0.707107 -0.707107 0 0"  gravcomp="{gravity_compensation}">
                        <inertial pos="-3.141e-03 -2.872e-02  3.495e-03" mass="0.646926" fullinertia="7.9620e-03 2.8110e-02 2.5995e-02 -3.9250e-03 1.0254e-02 7.0400e-04" />
                        <joint name="panda_joint2" range="-1.7628 1.7628" />
                        <geom class="visual"    mesh="link2_visual" material="white"/>
                        <geom class="collision" mesh="link2_collision" />

                        <body name="panda_link3" pos="0 -0.316 0" quat="0.707107 0.707107 0 0" gravcomp="{gravity_compensation}">
                            <inertial pos="2.7518e-02 3.9252e-02 -6.6502e-02" mass="3.228604" fullinertia="3.7242e-02 3.6155e-02 1.0830e-02 -4.7610e-03 -1.1396e-02 -1.2805e-02" />
                            <joint name="panda_joint3" range="-2.8973 2.8973" />
                            <geom class="visual"    mesh="link3_0_visual"  material="white"/>
                            <geom class="visual"    mesh="link3_1_visual"  material="white"/>
                            <geom class="visual"    mesh="link3_2_visual"  material="white"/>
                            <geom class="visual"    mesh="link3_3_visual"  material="black"/>
                            <geom class="collision" mesh="link3_collision" />

                            <body name="panda_link4" pos="0.0825 0 0" quat="0.707107 0.707107 0 0" gravcomp="{gravity_compensation}">
                                <inertial pos="-5.317e-02 1.04419e-01 2.7454e-02" mass="3.587895" fullinertia="2.5853e-02 1.9552e-02 2.8323e-02 7.7960e-03 -1.3320e-03 8.6410e-03" />
                                <joint name="panda_joint4" range="-3.0718 -0.0698" />
                                <geom class="visual"    mesh="link4_0_visual"  material="white"/>
                                <geom class="visual"    mesh="link4_1_visual"  material="white"/>
                                <geom class="visual"    mesh="link4_2_visual"  material="black"/>
                                <geom class="visual"    mesh="link4_3_visual"  material="white"/>
                                <geom class="collision" mesh="link4_collision" />

                                <body name="panda_link5" pos="-0.0825 0.384 0" quat="0.707107 -0.707107 0 0" gravcomp="{gravity_compensation}">
                                    <inertial pos="-1.1953e-02 4.1065e-02 -3.8437e-02" mass="1.225946" fullinertia="3.5549e-02 2.9474e-02 8.6270e-03 -2.1170e-03 -4.0370e-03 2.2900e-04" />
                                    <joint name="panda_joint5" range="-2.8973 2.8973"  />
                                    <geom class="visual"    mesh="link5_0_visual"  material="black"/>
                                    <geom class="visual"    mesh="link5_1_visual"  material="white"/>
                                    <geom class="visual"    mesh="link5_2_visual"  material="white"/>
                                    <geom class="collision" mesh="link5_0_collision" />
                                    <geom class="collision" mesh="link5_1_collision" />
                                    <geom class="collision" mesh="link5_2_collision" />

                                    <body name="panda_link6" pos="0 0 0" quat="0.707107 0.707107 0 0" gravcomp="{gravity_compensation}">
                                        <inertial pos="6.0149e-02 -1.4117e-02 -1.0517e-02" mass="1.666555" fullinertia="1.9640e-03 4.3540e-03 5.4330e-03 1.0900e-04 -1.1580e-03 3.4100e-04" />
                                        <joint name="panda_joint6" range="-0.0175 3.7525" />
                                        <geom class="visual"    mesh="link6_0_visual"  material="off_white"/>
                                        <geom class="visual"    mesh="link6_1_visual"  material="white"/>
                                        <geom class="visual"    mesh="link6_2_visual"  material="black"/>
                                        <geom class="visual"    mesh="link6_3_visual"  material="white"/>
                                        <geom class="visual"    mesh="link6_4_visual"  material="white"/>
                                        <geom class="visual"    mesh="link6_5_visual"  material="white"/>
                                        <geom class="visual"    mesh="link6_6_visual"  material="white"/>
                                        <geom class="visual"    mesh="link6_7_visual"  material="light_blue"/>
                                        <geom class="visual"    mesh="link6_8_visual"  material="light_blue"/>
                                        <geom class="visual"    mesh="link6_9_visual"  material="black"/>
                                        <geom class="visual"    mesh="link6_10_visual" material="black"/>
                                        <geom class="visual"    mesh="link6_11_visual" material="white"/>
                                        <geom class="visual"    mesh="link6_12_visual" material="green"/>
                                        <geom class="visual"    mesh="link6_13_visual" material="white"/>
                                        <geom class="visual"    mesh="link6_14_visual" material="black"/>
                                        <geom class="visual"    mesh="link6_15_visual" material="black"/>
                                        <geom class="visual"    mesh="link6_16_visual" material="white"/>
                                        <geom class="collision" mesh="link6_collision" />

                                        <body name="panda_link7" pos="0.088 0 0" quat="0.707107 0.707107 0 0" gravcomp="{gravity_compensation}">
                                            <inertial pos="1.0517e-02 -4.252e-03 6.1597e-02" mass="7.35522e-01" fullinertia="1.2516e-02 1.0027e-02 4.8150e-03 -4.2800e-04 -1.1960e-03 -7.4100e-04" />
                                            <joint name="panda_joint7" range="-2.8973 2.8973" />
                                            <geom class="visual"    mesh="link7_0_visual"  material="white"/>
                                            <geom class="visual"    mesh="link7_1_visual"  material="black"/>
                                            <geom class="visual"    mesh="link7_2_visual"  material="black"/>
                                            <geom class="visual"    mesh="link7_3_visual"  material="black"/>
                                            <geom class="visual"    mesh="link7_4_visual"  material="black"/>
                                            <geom class="visual"    mesh="link7_5_visual"  material="black"/>
                                            <geom class="visual"    mesh="link7_6_visual"  material="black"/>
                                            <geom class="visual"    mesh="link7_7_visual"  material="white"/>
                                            <geom class="collision" mesh="link7_collision" />

                                            {xml_ee}
                                        </body>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </worldbody>
        <contact>
            <exclude body1="panda_link0" body2="panda_link1" />
            <exclude body1="panda_link0" body2="panda_link2" />
            <exclude body1="panda_link0" body2="panda_link3" />
            <exclude body1="panda_link0" body2="panda_link4" />

            <exclude body1="panda_link1" body2="panda_link2" />
            <exclude body1="panda_link1" body2="panda_link4" />

            <exclude body1="panda_link2" body2="panda_link3" />
            <exclude body1="panda_link2" body2="panda_link4" />

            <exclude body1="panda_link3" body2="panda_link4" />
            <exclude body1="panda_link3" body2="panda_link6" />

            <exclude body1="panda_link4" body2="panda_link5" />
            <exclude body1="panda_link4" body2="panda_link6" />
            <exclude body1="panda_link4" body2="panda_link7" />

            <exclude body1="panda_link5" body2="panda_link7" />

            <exclude body1="panda_link6" body2="panda_link7" />
            {xml_exclude}
        </contact>

        {xml_touch_sensor}
    """
    return xml_robot

def generate_model_xml_string(
                            obj_type: str="box",
                            obj_xy_pos: np.ndarray=np.array([0.3, -0.1]),
                            target_xy_pos: np.ndarray=np.array([0.5, 0.1]),
                            obj_quat: np.ndarray=np.array([1,0,0,0]),
                            target_quat: np.ndarray=np.array([1,0,0,0]),
                            obj_size_0: float=0.055,
                            obj_size_1: float=0.055,
                            obj_height: float=0.04,
                            obj_mass: float=0.2,
                            obj_sliding_fric: float=0.8,
                            obj_torsional_fric: float=0.005,
                            robot_gravity_compensation: float=1,
                            fingertip_model: bool=False,
                            use_red_ee: bool=False,
                            path_to_assets: str=os.path.join(panda_push_mujoco.__path__[0], "assets"),
                            include_camera: bool=True, 
                            include_actuator: bool=True,
                            use_sim_config: bool = True):
    
    # camera
    if include_camera:
        if use_sim_config:
            camera_pos = [0.4, 0.0, -0.1]
            camera_z_angle = 0.0
            camera_fovy = 65
        else:
            camera_pos = [-0.08, 0.552, 0.19]
            camera_z_angle = 0
            camera_fovy = 117
        xml_camera = fr"""
            <!-- camera -->
            <body name="camera_link" pos="{camera_pos[0]} {camera_pos[1]} {camera_pos[2]}" euler="0 0 {camera_z_angle}">
                <camera name="rgb_cam" mode="fixed" fovy="{camera_fovy}" euler="3.141 0 0"/>
                <geom name="cam_geom" pos="0 0 0" type="box" size="0.05 0.05 0.05" rgba="0.8 0 0 0.2" group="0"/>
            </body> 
        """
    else:
        xml_camera = ""

    # actuator
    if include_actuator:
        xml_actuator = r"""
        <actuator>
            <velocity name="v_servo_panda_joint1" joint="panda_joint1" kv="30" />
            <velocity name="v_servo_panda_joint2" joint="panda_joint2" kv="30" />
            <velocity name="v_servo_panda_joint3" joint="panda_joint3" kv="30" />
            <velocity name="v_servo_panda_joint4" joint="panda_joint4" kv="30" />
            <velocity name="v_servo_panda_joint5" joint="panda_joint5" kv="10" />
            <velocity name="v_servo_panda_joint6" joint="panda_joint6" kv="10" />
            <velocity name="v_servo_panda_joint7" joint="panda_joint7" kv="5" />
        </actuator>
        """
    else:
        xml_actuator = ""

    # robot
    xml_robot = get_panda_robot_xml_str(gravity_compensation=robot_gravity_compensation,
                                        path_to_meshes=os.path.join(path_to_assets, "meshes"),
                                        fingertip_model=fingertip_model,
                                        use_red_ee=use_red_ee)

    if obj_type == "cylinder":
        geom_size_str = f"{obj_size_0} {obj_height}"
    else:
        geom_size_str = f"{obj_size_0} {obj_size_1} {obj_height}"
        
    if use_sim_config:
        table_height = 0.4
        table_pos = [0.525, 0.0, table_height/2]
        table_size = [0.35, 0.45, table_height/2]
    else:
        table_height = 0.385
        table_pos = [0.0, 0.595, table_height/2]
        table_size = [0.45, 0.35 + 0.06, table_height/2]

    texture_path = os.path.join(path_to_assets, "textures")
    xml = fr"""
    <mujoco model="panda">
        <compiler angle="radian" coordinate="local" texturedir="{texture_path}" />
        <option timestep="0.001" cone="elliptic" jacobian="sparse" gravity="0 0 -9.81"/>

        {xml_robot}

        <asset>
            <texture builtin="flat" height="512" width="512" name="texplane" rgb1=".51 .51 .51" rgb2=".51 .51 .51" />
            <texture type="skybox" builtin="gradient" rgb1="0.8 0.898 1" rgb2="0.8 0.898 1" width="32" height="32" />
            <texture name="texture_object" file="block_blue.png" gridsize="3 4" gridlayout=".U..LFRB.D.." />

            <material name="floor_mat" reflectance="0.5" shininess="0.01" specular="0.1" texture="texplane" texuniform="true" />
            <material name="table_mat" specular="0.1" shininess="0.5" reflectance="0.5" rgba="0.73 0.73 0.73 1.0" />
            <material name="object_mat" specular="0" shininess="0.5" reflectance="0" texture="texture_object" />
        </asset>

        <worldbody>
            <geom name="ground_plane" pos="0.4 0. 0" type="plane" conaffinity="3" condim="3" size="2 2 10" material="floor_mat" group="2"/>
            <geom name="table" pos="{table_pos[0]} {table_pos[1]} {table_pos[2]}" size="{table_size[0]} {table_size[1]} {table_size[2]}" conaffinity="3" type="box" material="table_mat" friction="{obj_sliding_fric} {obj_torsional_fric} 0.0001" mass="200" group="2"/>

            <body name="target" pos="{target_xy_pos[0]} {target_xy_pos[1]} {table_height + obj_height}" quat="{target_quat[0]} {target_quat[1]} {target_quat[2]} {target_quat[3]}">
                <joint name="target_joint" type="free" damping="0.01"></joint>
                <geom size="{geom_size_str}" type="{obj_type}" contype="2" conaffinity="2" name="target_geom" rgba="0 1 0 1" mass="{obj_mass}" friction="{obj_sliding_fric} {obj_torsional_fric} 0.0001" group="2" />
            </body>

            <body name="object" pos="{obj_xy_pos[0]} {obj_xy_pos[1]} {table_height + obj_height}" quat="{obj_quat[0]} {obj_quat[1]} {obj_quat[2]} {obj_quat[3]}">
                <joint name="object_joint" type="free" damping="0.01"></joint>
                <geom size="{geom_size_str}" type="{obj_type}" condim="3" contype="1" conaffinity="1" name="object_geom" material="object_mat" mass="{obj_mass}" friction="{obj_sliding_fric} {obj_torsional_fric} 0.0001" group="1"/>
                <site name="object_site" pos="0 0 0" size="0.001" rgba="1 0 0 0.5" type="sphere" />
            </body>

            {xml_camera}

            <light directional="true" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="false" pos="0 0 4" dir="0 0 -1" name="light0" />
        </worldbody>

        <sensor>
            <framepos name="object_pos" objtype="geom" objname="object_geom"/>
            <framequat name="object_quat" objtype="geom" objname="object_geom"/>
        </sensor>

        
        {xml_actuator}
    </mujoco>
    """

    return xml

class MuJoCoPandaPushController():

    def __init__(self, 
                 model: MjModel, 
                 robot_name: str, 
                 ee_site_name: str, 
                 initial_ee_zpos: float, 
                 min_ee_xy_pos: np.ndarray = np.array([0.1, -0.25]),
                 max_ee_xy_pos: np.ndarray = np.array([0.65, 0.25]),
                 use_sim_config: bool = True,
                 safety_dq_scale: float = 1.0):
        self.robot_name = robot_name
        # EE site and initial zPos
        self.ee_site_name = ee_site_name
        self.initial_ee_zpos = initial_ee_zpos

        self.load_model_data(model)

        # min max x,y EE pos
        self.min_ee_xy_pos = min_ee_xy_pos
        self.max_ee_xy_pos = max_ee_xy_pos

        # position and velocity limits for Panda robot
        # max, min joint velocities
        self.dq_max = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
        self.dq_min = (-1)*self.dq_max

        # max, min joint positions
        self.q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        self.q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])

        # controller config (sim vs. real)
        self.use_sim_config = use_sim_config

        # safety velocity scale 
        self.safety_dq_scale = safety_dq_scale

    def load_model_data(self, model):
        # remember robot joint names, ids, qposadr and dofadr
        self.panda_joint_names, self.panda_joint_ids, self.panda_joint_qposadr, self.panda_joint_dofadr = get_model_robot_joints(model, self.robot_name)
        # actuator names and ids
        self.panda_actuator_names, self.panda_actuator_ids = get_model_actuators(model)
        # EE site 
        self.ee_site_id = model.site(self.ee_site_name).id

    def update_desired_pose(self, model: MjModel, data: MjData, action: np.ndarray):
        assert action.shape[0] == 2 or action.shape[0] == 3

        # desired EE pos
        ee_pos = get_site_xpos(model, data, self.ee_site_name)
        self.ee_pos_d = np.array([ee_pos[0] + action[0], ee_pos[1] + action[1], self.initial_ee_zpos])
        self.ee_pos_d[:2] = np.minimum(np.maximum(self.ee_pos_d[:2], self.min_ee_xy_pos), self.max_ee_xy_pos)

        if action.shape[0] == 3:
            ee_rotmat = data.site_xmat[self.ee_site_id,:].reshape(3, 3)
            # align x-axis of base frame with x-axis of ee frame
            self.cone_ref_vec_x = np.array([[1,0,0]])
            ee_x_axis = ee_rotmat[:,0]
            # angle between x-axis (base frame) and projection of x-axis into xy-plane of base frame
            current_angle = np.arctan2(ee_x_axis[1], ee_x_axis[0])
            desired_angle = rotations_utils.add_normalized_angles(current_angle, action[-1])
            self.cone_ref_vec_x = np.array([[np.cos(desired_angle),np.sin(desired_angle),0]])

    def update(self, model: MjModel, data: MjData, action: np.ndarray):

        # get Jacobians and EE pose
        ee_rotmat = data.site_xmat[self.ee_site_id,:].reshape(3, 3)
        ee_pos = get_site_xpos(model, data, self.ee_site_name)
        jacp = get_site_jacp(model, data, self.ee_site_id)[:, self.panda_joint_dofadr]
        jacr = get_site_jacr(model, data, self.ee_site_id)[:, self.panda_joint_dofadr]

        # cone and position error
        task_size = 4 + int(action.shape[0] == 3)
        conepos_error = np.zeros(task_size)

        # align z-axis of ee frame with z-axis*(-1) of base frame
        cone_ref_vec_z = np.array([[0,0,-1]])
        ee_z_axis = ee_rotmat[:,2]
       
        if action.shape[0] == 3:
            conepos_error[0] = (self.cone_ref_vec_x @ self.ee_x_axis.reshape(-1,1) - 1) # error cone task (x-axis)

        idx_cone_z_task = int(action.shape[0] == 3)
        conepos_error[idx_cone_z_task] = (cone_ref_vec_z @ ee_z_axis.reshape(-1,1) - 1) # error cone task (z-axis)
        conepos_error[idx_cone_z_task+1:] = self.ee_pos_d - ee_pos # error position task

        # cone and position task
        jac_conepos = np.zeros((task_size,7))

        if action.shape[0] == 3:
            ee_x_axis = ee_rotmat[:,0]
            jac_conepos[0,6] = self.cone_ref_vec_x @ controller_utils.vec2SkewSymmetricMat(ee_x_axis) @ jacr[:,6]
        jac_conepos[idx_cone_z_task, :6] = cone_ref_vec_z @ controller_utils.vec2SkewSymmetricMat(ee_z_axis) @ jacr[:, :6]
        jac_conepos[idx_cone_z_task + 1:, :6] = jacp[:,:6].copy()

        jac_conepos_pinv = controller_utils.pinv(jac_conepos, use_damping=self.use_sim_config)
        dq_d = jac_conepos_pinv @ conepos_error

        # set desired joint velocities and ensure joint position and velocity limits
        data.ctrl[self.panda_actuator_ids] = self.ensure_joint_pos_velo_limits(data, dq_d)
    
    def ensure_joint_pos_velo_limits(self, data: MjData, dq: np.ndarray):
        q = data.qpos[self.panda_joint_qposadr]

        if self.use_sim_config:
            dq = self.safety_dq_scale * np.clip(dq, a_min=np.maximum(self.dq_min, self.q_min - q), a_max=np.minimum(self.dq_max, self.q_max - q))
        else:
            pos_aware_dq_min = np.maximum(self.dq_min, self.q_min - q)
            pos_aware_dq_max = np.minimum(self.dq_max, self.q_max - q)
            # uniformly scale qdot if any limit is exceeded
            scales = np.maximum(dq / pos_aware_dq_min, dq / pos_aware_dq_max)
            scales[np.logical_not(np.isfinite(scales))] = 1.0
            dq = (self.safety_dq_scale * dq) / np.maximum(1.0, np.max(scales))

        assert (dq <= self.dq_max).all()

        return dq