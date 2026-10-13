from glob import glob
from setuptools import find_packages, setup


package_name = "huanpan_mujoco"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="lzt",
    maintainer_email="lzt@example.com",
    description="MuJoCo trajectory bridge for huanpan_ws.",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "mujoco_bridge = huanpan_mujoco.bridge_node:main",
            "smoke_trajectory = huanpan_mujoco.smoke_trajectory:main",
        ],
    },
)
