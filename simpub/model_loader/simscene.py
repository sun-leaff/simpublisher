import io
import math
import random 
from pathlib import Path
import numpy as np
from hashlib import md5

from simpub import mjcf

from simpub.simdata import *
from simpub.model_loader.asset_loader import MeshLoader, TextureLoader

from simpub.transform import mj2euler, mj2pos, mj2scale, quat2euler
from scipy.spatial.transform import Rotation as Rot

class SimScene:
  def __init__(self) -> None:

    self.id = str(random.randint(int(1e9), int(1e10 - 1))) # [100.000.000, 999.999.999]
    
    self._worldbody : SimBody = None
    self._assets : dict[SimAssetType, dict[str, SimAsset]] = dict()

    for type in SimAssetType:
      self.assets[type] = dict()

    self.xml_string : str
    self.xml_assets : dict

  @staticmethod
  def from_file(path : Path | str) -> "SimScene":

    file_path : Path = path if isinstance(path, Path) else Path(path)
    file_path = file_path.absolute() 

    scene = SimScene()
    
    data : mjcf.RootElement = mjcf.from_path(file_path, resolve_references=True)
    scene._load_mjcf(data)

    return scene

  @staticmethod
  def from_string(content : str, assets : dict) -> "SimScene":
    scene = SimScene()
    
    data : mjcf.RootElement = mjcf.from_xml_string(content, assets=assets)
    scene._load_mjcf(data)

    return scene
  
  @property
  def worldbody(self) -> SimBody:
    return self._worldbody
      

  @property
  def assets(self) -> dict[SimAssetType, dict[str, SimAsset]]:
    return self._assets
  
  """
  MJCF file loading with dm_control.mjcf
  """
  def _load_mjcf(self, data : mjcf.RootElement):
    
    
    eulerseq = data.compiler.eulerseq or "xyz"
    angle_type = data.compiler.angle or "degrees"

    angle_fn = lambda x: x if angle_type =="degrees" else np.rad2deg(x)
 


    self.xml_string = data.to_xml_string()
    self.xml_assets = data.get_assets()

    # commit all the default values
    for tag in { "geom", "mesh", "joint", "body", "material"}: # Add whatever you are using
      for elem in data.find_all(tag):
        if not hasattr(elem, "dclass"): continue  
        mjcf.commit_defaults(elem)    

    
    for child in data.asset.all_children():
      match child.tag:
        case "mesh":
          mesh = MeshLoader.fromBytes(
            child.name or child.file.prefix, 
            child.file.contents, 
            mesh_type=child.file.extension[1:], 
            scale=child.scale
          )
          self._assets[SimAssetType.MESH][mesh.tag] = mesh
        case "material":
          color = child.rgba if child.rgba is not None else np.array([1, 1, 1, 1])
          asset = SimMaterial(
            tag=child.name,
            color=color,
            emissionColor=((child.emission or 0.0) * color),   
            specular=child.specular or 0.5,
            shininess=child.shininess or 0.5,
            reflectance=child.reflectance or 0.0
          )

          if child.texture: 
            asset.texture = child.texture.name
          
          if child.texrepeat is not None:
            asset.texsize = child.texrepeat

          self._assets[SimAssetType.MATERIAL][asset.tag] = asset

        case "texture":
          if child.builtin != "none" and child.builtin is not None:
            texture = TextureLoader.fromBuiltin(child.name or child.type, child.builtin, child.rgb1)
          else:
            texture = TextureLoader.fromBytes(child.name or child.type, child.file.contents, child.type or "cube", child.rgb1)

          self._assets[SimAssetType.TEXTURE][texture.tag] = texture

    def rotation_from_object(obj : mjcf.Element) -> np.ndarray: # https://mujoco.readthedocs.io/en/stable/modeling.html#frame-orientations

      quat = getattr(obj, "quat", None)
      axisangle = getattr(obj, "axisangle", None)
      euler = getattr(obj, "euler", None)
      xyaxes = getattr(obj, "xyaxes", None)
      zaxis = getattr(obj, "zaxis", None)

      result : np.ndarray
      if quat is not None:
        result = quat2euler(quat)
      elif axisangle is not None:
        raise NotImplementedError()
      elif euler is not None:
        match eulerseq:
          case "xyz":
            result = euler
          case "zyx":
            result = euler[::-1]
      elif xyaxes is not None:
        x = xyaxes[:3]
        y = xyaxes[3:6]
        z = np.cross(x, y)
        result = Rot.from_matrix(np.array([x, y, z]).T).as_euler("xyz")
      elif zaxis is not None:
        raise NotImplementedError()
      else:
        result = np.array([0, 0, 0])

      return mj2euler(angle_fn(result))
      

  
    def load_visual(visual : mjcf.Element) -> Optional[SimVisual]:    
      if visual.group is not None and visual.group > 2: return None # find a better way to figure this out 


      type = SimVisualType(visual.type.upper()) if visual.type else SimVisualType.SPHERE

      transform = SimTransform(
        position=mj2pos(visual.pos), 
        rotation=rotation_from_object(visual), 
        scale=np.abs(mj2scale(visual.size))
      )

      return SimVisual(
        name=visual.name or visual.tag,
        type=type,
        transform=transform,
        asset=(visual.mesh.name or visual.mesh.file.prefix) if type == SimVisualType.MESH else None, # this is a unique file id specified by mujoco
        material=visual.material.name if visual.material else None,
        color=visual.rgba if visual.rgba is not None else np.array([1, 1, 1, 1])
      )
    
    def load_joint(root : mjcf.Element):
      
      ujoint =  SimJoint(
        name=root.name or root.tag,
        body=load_body(root),
        transform=SimTransform(
          position=mj2pos(root.pos), 
          rotation=rotation_from_object(root)
        )
      )

      joints = root.get_children("joint")
      freejoint = root.get_children("freejoint")

      # TODO: There could be multiple joints attached to each body, chaining them together should work 
      # https://mujoco.readthedocs.io/en/stable/modeling.html#kinematic-tree
      if freejoint is not None:
        joint = freejoint
        ujoint.type = SimJointType.FREE
      elif len(joints) > 0: 
        joint = joints[0]
        ujoint.name = joint.name or ujoint.name
        ujoint.transform.position += mj2pos(joint.pos)
        ujoint.transform.rotation += rotation_from_object(joint)

        if joint.range is not None:
          ujoint.minrot = math.degrees(joint.range[0])
          ujoint.maxrot = math.degrees(joint.range[1])
      
        ujoint.type = SimJointType((joint.type or "hinge").upper())
        ujoint.axis = mj2pos(joint.axis)

      return ujoint

    def load_body(body : mjcf.Element) -> SimBody:
      return SimBody(
        name=body.name if hasattr(body, "name") else "worldbody",
        joints=[load_joint(b) for b in body.body],
        visuals=[visual for geom in body.get_children("geom") if (visual := load_visual(geom)) is not None], # this is hacky is there a better way to detect if a geom is collision or visual 
      )
    
    self._worldbody = load_body(data.worldbody)

