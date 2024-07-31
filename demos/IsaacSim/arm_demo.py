# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import io
from hashlib import md5
import trimesh
import random

from simpub.server import SimPublisher
from simpub.simdata import (
    SimScene,
    SimObject,
    SimVisual,
    VisualType,
    SimTransform,
    SimMesh,
)

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates different single-arm manipulators."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import omni.usd
from pxr import Usd, UsdGeom, UsdUtils, UsdPhysics
import numpy as np
import omni
import carb
from omni.physx.scripts import utils

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

print("=" * 20)
print(ISAACLAB_NUCLEUS_DIR)

##
# Pre-defined configs
##
# isort: off
from omni.isaac.lab_assets import (
    FRANKA_PANDA_CFG,
    UR10_CFG,
    KINOVA_JACO2_N7S300_CFG,
    KINOVA_JACO2_N6S300_CFG,
    KINOVA_GEN3_N7_CFG,
    SAWYER_CFG,
)

# isort: on


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""
    # currently this function is unimportant, since we only test with a single origin/env.

    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_rows = np.floor(np.sqrt(num_origins))
    num_cols = np.ceil(num_origins / num_rows)
    xx, yy = torch.meshgrid(
        torch.arange(num_rows), torch.arange(num_cols), indexing="xy"
    )
    env_origins[:, 0] = (
        spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    )
    env_origins[:, 1] = (
        spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    )
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # this function will build the scene by adding primitives to it.

    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a mount and a robot on top of it
    origins = define_origins(num_origins=1, spacing=2.0)

    # Origin 1 with Franka Panda
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    # -- Table
    cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    )
    cfg.func("/World/Origin1/Table", cfg, translation=(0.55, 0.0, 1.05))
    cfg.func("/World/Origin1/Table_1", cfg, translation=(0.55, 3.0, 1.05))

    # -- Robot
    franka_arm_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Origin1/Robot")
    franka_arm_cfg.init_state.pos = (0.0, 0.0, 1.05)
    franka_panda = Articulation(cfg=franka_arm_cfg)

    # -- cube
    cfg_cube = sim_utils.CuboidCfg(
        size=(0.1, 0.1, 0.1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
    )
    cfg_cube.func("/World/Origin1/Cube1", cfg_cube, translation=(0.2, 0.0, 3.0))

    # return the scene information
    scene_entities = {
        "franka_panda": franka_panda,
    }
    return scene_entities, origins


def run_simulator(
    sim: sim_utils.SimulationContext,
    entities: dict[str, Articulation],
    origins: torch.Tensor,
):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset the scene entities
            for index, robot in enumerate(entities.values()):
                # root state
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_state_to_sim(root_state)
                # set joint positions
                joint_pos, joint_vel = (
                    robot.data.default_joint_pos.clone(),
                    robot.data.default_joint_vel.clone(),
                )
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                # clear internal buffers
                robot.reset()
            print("[INFO]: Resetting robots state...")
        # apply random actions to the robots
        for robot in entities.values():
            # generate random joint positions
            joint_pos_target = (
                robot.data.default_joint_pos
                + torch.randn_like(robot.data.joint_pos) * 0.1
            )
            joint_pos_target = joint_pos_target.clamp_(
                robot.data.soft_joint_pos_limits[..., 0],
                robot.data.soft_joint_pos_limits[..., 1],
            )
            # apply action to the robot
            robot.set_joint_position_target(joint_pos_target)
            # write data to sim
            robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)


class IsaacSimPublisher(SimPublisher):
    def __init__(self, host: str, stage: Usd.Stage) -> None:
        sim_scene = self.parse_scene(stage)
        super().__init__(sim_scene, host)

    def parse_scene(self, stage: Usd.Stage) -> SimScene:
        print("=" * 50)
        print("parsing stage:", stage)

        scene = SimScene()
        self.sim_scene = scene

        scene.root = SimObject(name="root")

        obj1 = SimObject(name="object_1")
        obj1.visuals.append(
            SimVisual(
                type=VisualType.CUBE,
                color=[0.5, 0.7, 0.6, 1.0],
                trans=SimTransform(),
            )
        )
        scene.root.children.append(obj1)

        bin_buffer = io.BytesIO()

        mesh = trimesh.creation.box(extents=[1, 2, 3])

        indices = mesh.faces.astype(np.int32)
        bin_buffer = io.BytesIO()

        # Vertices
        verts = mesh.vertices.astype(np.float32)
        verts[:, 2] = -verts[:, 2]
        verts = verts.flatten()
        vertices_layout = bin_buffer.tell(), verts.shape[0]
        bin_buffer.write(verts)

        # Normals
        norms = mesh.vertex_normals.astype(np.float32)
        norms[:, 2] = -norms[:, 2]
        norms = norms.flatten()
        normal_layout = bin_buffer.tell(), norms.shape[0]
        bin_buffer.write(norms)

        # Indices
        indices = mesh.faces.astype(np.int32)
        indices = indices[:, [2, 1, 0]]
        indices = indices.flatten()
        indices_layout = bin_buffer.tell(), indices.shape[0]
        bin_buffer.write(indices)

        # # Texture coords
        # uv_layout = (0, 0)
        # if hasattr(mesh.visual, "uv"):
        #     uvs = mesh.visual.uv.astype(np.float32)
        #     uvs[:, 1] = 1 - uvs[:, 1]
        #     uvs = uvs.flatten()
        #     uv_layout = bin_buffer.tell(), uvs.shape[0]

        bin_data = bin_buffer.getvalue()
        hash = md5(bin_data).hexdigest()

        mesh = SimMesh(
            id="mesh_1",
            indicesLayout=indices_layout,
            verticesLayout=vertices_layout,
            dataHash=hash,
            normalsLayout=normal_layout,
            uvLayout=(0, 0),
        )

        self.sim_scene.meshes.append(mesh)
        self.sim_scene.raw_data[mesh.dataHash] = bin_data

        obj2 = SimObject(name="object_2")
        obj2.visuals.append(
            SimVisual(
                type=VisualType.MESH,
                mesh="mesh_1",
                color=[0.5, 0.7, 0.6, 1.0],
                trans=SimTransform(pos=[5, 0, 0]),
            )
        )
        scene.root.children.append(obj2)

        obj2 = self.parse_prim_tree(stage.GetPrimAtPath("/World/Origin1/Robot"))
        assert obj2 is not None
        scene.root.children.append(obj2)

        return scene

    def parse_prim_tree(self, root: Usd.Prim, indent=0) -> SimObject | None:
        if root.GetTypeName() not in {"Xform", "Mesh"}:  # Cube
            return

        purpose_attr = root.GetAttribute("purpose")
        if purpose_attr and purpose_attr.Get() in {"proxy", "guide"}:
            return

        trans_mat = omni.usd.get_local_transform_matrix(root)

        translate = trans_mat.ExtractTranslation()
        translate = [translate[0], translate[1], translate[2]]

        rot = trans_mat.ExtractRotationQuat()
        imag = rot.GetImaginary()
        rot = [imag[0], imag[1], imag[2], rot.GetReal()]

        sim_object = SimObject(
            name=str(root.GetPrimPath()).replace("/", "_"),
            trans=SimTransform(pos=translate, rot=rot),
        )

        carb.log_info(
            "\t" * indent
            + f"{root.GetName()}: {root.GetTypeName()} {root.GetAttribute('purpose').Get()}"
        )

        # maybe time_code is necessary
        # trans_mat = omni.usd.get_local_transform_matrix(root)
        # carb.log_info("\t" * indent + f"{trans_mat}")

        # attr: Usd.Property
        # for attr in root.GetProperties():
        #     carb.log_info("\t" * indent + f"{attr.GetName()}")

        if root.GetTypeName() == "Mesh":
            mesh_prim = UsdGeom.Mesh(root)
            assert mesh_prim is not None

            points = (
                np.asarray(mesh_prim.GetPointsAttr().Get()).astype(np.float32).flatten()
            )
            normals = (
                np.asarray(mesh_prim.GetNormalsAttr().Get())
                .astype(np.float32)
                .flatten()
            )
            indices = (
                np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
                .astype(np.int32)
                .flatten()
            )

            carb.log_info(f"normals: {normals.shape}")
            # carb.log_info(
            #     "\t" * indent + f"@vert {points} {points.dtype} {points.shape}"
            # )
            # carb.log_info(
            #     "\t" * indent + f"@indi {indices} {indices.dtype} {indices.shape}"
            # )

            bin_buffer = io.BytesIO()

            # Vertices
            vertices_layout = bin_buffer.tell(), points.shape[0]
            bin_buffer.write(points)

            normals_layout = bin_buffer.tell(), normals.shape[0]
            bin_buffer.write(normals)

            # Indices
            indices_layout = bin_buffer.tell(), indices.shape[0]
            bin_buffer.write(indices)

            bin_data = bin_buffer.getvalue()
            hash = md5(bin_data).hexdigest()

            mesh_id = "@mesh-" + str(random.randint(int(1e9), int(1e10 - 1)))
            mesh = SimMesh(
                id=mesh_id,
                indicesLayout=indices_layout,
                verticesLayout=vertices_layout,
                dataHash=hash,
                normalsLayout=normals_layout,
                uvLayout=(0, 0),
            )

            self.sim_scene.meshes.append(mesh)
            self.sim_scene.raw_data[mesh.dataHash] = bin_data

            sim_mesh = SimVisual(
                type=VisualType.MESH,
                mesh=mesh_id,
                color=[0.5, 0.7, 0.6, 1.0],
                trans=SimTransform(pos=[5, 0, 0]),
            )
            sim_object.visuals.append(sim_mesh)

        child: Usd.Prim

        if root.IsInstance():
            proto = root.GetPrototype()
            carb.log_info("\t" * indent + f"@prototype: {proto.GetName()}")

            for child in proto.GetChildren():
                if obj := self.parse_prim_tree(child, indent + 1):
                    sim_object.children.append(obj)

        else:
            for child in root.GetChildren():
                if obj := self.parse_prim_tree(child, indent + 1):
                    sim_object.children.append(obj)

        return sim_object

    def initialize_task(self):
        super().initialize_task()

    def get_update(self) -> dict[str, list[float]]:
        state = {"object_1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
        # for name, trans in self.tracked_obj_trans.items():
        #     pos, rot = trans
        #     state[name] = [-pos[1], pos[2], pos[0], rot[2], -rot[3], -rot[1], rot[0]]
        return state

    def shutdown(self):
        self.stream_task.shutdown()
        self.msg_service.shutdown()

        self.running = False
        self.thread.join()


def parse_stage(stage: Usd.Stage):
    publisher = IsaacSimPublisher(host="192.168.0.134", stage=stage)
    publisher.start()


def main():
    """Main function."""
    # sim_utils.SimulationContext is a singleton class
    # if SimulationContext.instance() is None:
    #     self.sim: SimulationContext = SimulationContext(self.cfg.sim)

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")
    parse_stage(sim.stage)

    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


# stage: Usd.Stage = omni.usd.get_context().get_stage()
# carb.log_info(f"stage: {stage}\n")


# # def iterate_prim_children(root: Usd.Prim):
# #     carb.log_info(f"{root.GetName()}")
# #     obj: Usd.Prim
# #     for obj in root.GetChildren():
# #         carb.log_info("")
# #         carb.log_info("=" * 50)
# #         carb.log_info(f"obj: {obj}")
# #         carb.log_info(f"type: {obj.GetTypeName()}")
# #         carb.log_info(f"is instance: {obj.IsInstance()}")
# #         carb.log_info(f"is instance proxy: {obj.IsInstanceProxy()}")
# #         carb.log_info(f"is instancable: {obj.IsInstanceable()}")
# #         if obj.IsInstance():
# #             carb.log_info(obj.GetPrototype().IsInPrototype())
# #             carb.log_info(obj.GetPrototype().GetChildrenNames())
# #             carb.log_info(obj.GetPrototype().GetTypeName())
# #             for child in obj.GetPrototype().GetChildren():
# #                 carb.log_info(f"{child.GetName()} {child.GetTypeName()}")
# #         else:
# #             carb.log_info(obj.GetChildrenNames())
# #             for i, name in enumerate(obj.GetPropertyNames()):
# #                 carb.log_info(name)

# #         if obj.IsInstance():
# #             iterate_prim_children(obj.GetPrototype())
# #         else:
# #             iterate_prim_children(obj)


# # iterate_prim_children(stage.GetPrimAtPath("/World/Origin1"))

# # timeline = omni.timeline.get_timeline_interface()
# # timecode = timeline.get_current_time() * timeline.get_time_codes_per_seconds()

# # prim = stage.GetPrimAtPath("/World/Origin1/Table/Visuals")

# # print(omni.usd.get_world_transform_matrix(prim, timecode))
# # carb.log_info(prim)
# # carb.log_info(prim.IsInstance())
# # carb.log_info(prim.GetTypeName())
# # carb.log_info(prim.GetPrototype())
# # proto = prim.GetPrototype()
# # carb.log_info(proto.GetChildrenNames())
# # geom = proto.GetChild("TableGeom")
# # carb.log_info(geom)
# # carb.log_info(geom.GetTypeName())
# # carb.log_info(geom.GetChildrenNames())
# # carb.log_info(geom.GetChild("subset").GetChildrenNames())
# # mesh = UsdGeom.Mesh(geom)
# # carb.log_info(mesh)
# # points = np.asarray(mesh.GetPointsAttr().Get())
# # indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
# # carb.log_info(f"{points} {points.dtype} {points.shape}")
# # carb.log_info(f"{indices} {indices.dtype} {indices.shape}")
