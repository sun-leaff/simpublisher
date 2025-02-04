from mujoco import mj_name2id, mjtObj
from typing import List, Dict
import numpy as np

from simpub.server import SimPublisher
from simpub.parser.mjcf import MJCFParser
from simpub.simdata import SimObject


class MujocoPublisher(SimPublisher):

    def __init__(
        self,
        mj_model,
        mj_data,
        mjcf_path: str,
        host: str = "127.0.0.1",
        no_rendered_objects: List[str] = None,
        no_tracked_objects: List[str] = None,
    ) -> None:
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.parser = MJCFParser(mjcf_path)
        sim_scene = self.parser.parse()
        self.tracked_obj_trans: Dict[str, np.ndarray] = dict()
        super().__init__(
            sim_scene,
            no_rendered_objects,
            no_tracked_objects,
            host,
        )
        for child in self.sim_scene.root.children:
            self.set_update_objects(child)

    def set_update_objects(self, obj: SimObject):
        if obj.name in self.no_tracked_objects:
            return
        body_id = mj_name2id(self.mj_model, mjtObj.mjOBJ_BODY, obj.name)
        pos = self.mj_data.xpos[body_id]
        rot = self.mj_data.xquat[body_id]
        trans = (pos, rot)
        self.tracked_obj_trans[obj.name] = trans
        for child in obj.children:
            self.set_update_objects(child)

    def initialize_task(self):
        super().initialize_task()

    def get_update(self) -> Dict[str, List[float]]:
        state = {}
        for name, trans in self.tracked_obj_trans.items():
            pos, rot = trans
            state[name] = [
                -pos[1], pos[2], pos[0], rot[2], -rot[3], -rot[1], rot[0]
            ]
        return state

    def shutdown(self):
        self.stream_task.shutdown()
        self.msg_service.shutdown()

        self.running = False
        self.thread.join()
