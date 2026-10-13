"""Offline tooling for the huanpan slot/perception module.

This package is deliberately ROS-independent: it drives the MuJoCo workcell
directly to build a labelled RGB-D dataset, then trains and evaluates a
detector. The ROS 2 node that consumes the trained model lives in
``src/huanpan_perception`` (added later).
"""

__all__ = ["camera", "scene", "labels", "generate", "train", "evaluate"]
