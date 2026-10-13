from setuptools import find_packages, setup

package_name = "huanpan_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/perception.yaml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/perception.launch.py",
                "launch/capture.launch.py",
                "launch/fine_localization.launch.py",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="lzt",
    maintainer_email="lzt@example.com",
    description=(
        "YOLO-based rack/slot detection on the real Intel RealSense D435, "
        "plus dataset capture, labelling, training and evaluation tooling."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "detector_node = huanpan_perception.detector_node:main",
            "capture_node = huanpan_perception.capture_node:main",
            "fine_localization_node = huanpan_perception.fine_localization_node:main",
            "fine_localize_images = huanpan_perception.fine_localize_cli:main",
            "generate_tag_board = huanpan_perception.tag_board_cli:main",
            "fake_camera_node = huanpan_perception.fake_camera_node:main",
            "detect_images = huanpan_perception.detect_cli:main",
            "extract_frames = huanpan_perception.extract_cli:main",
            "analyze_depth = huanpan_perception.analyze_depth:main",
            "train_detector = huanpan_perception.train_cli:main",
            "evaluate_detector = huanpan_perception.eval_cli:main",
            "label_dataset = huanpan_perception.label_cli:main",
        ],
    },
)
