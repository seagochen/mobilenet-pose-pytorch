from setuptools import setup, find_packages

setup(
    name="lite-pose",
    version="0.1.0",
    description="Lightweight Multi-Person Pose Estimation with MobileNetV3 Backbone",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "onnxruntime>=1.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.14.0",
    ],
)
