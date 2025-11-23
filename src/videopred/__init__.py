# /data1/sijiaqi/instructpix2pix/src/videopred/__init__.py
from .model.wrap_model import WrapInstructPix2Pix
from .model.motion_encoder import MotionEncoder

__all__ = ["WrapInstructPix2Pix", "MotionEncoder"]