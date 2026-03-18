from setuptools import setup, find_packages

setup(
    name="customvla",
    version="0.1.0",
    description="Modular Vision-Language-Action package for robotic manipulation",
    author="VatsalyaBhadaurya",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy",
        "pillow",
        "tqdm",
        "pandas",
        "pyarrow",
        "av",
        "huggingface_hub",
        "ultralytics",      # YOLOv8 (from your original vla_object_tokenizer)
        "opencv-python",
    ],
    extras_require={
        "ros": ["rospy", "sensor_msgs"],
        "franka": ["frankx"],
        "ur": ["ur_rtde"],
    },
)
