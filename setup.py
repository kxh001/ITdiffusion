from setuptools import setup

setup(
    name="itdiffusion",
    py_modules=["utilsitd"],
    install_requires=[
        "blobfile>=1.0.5",
        "torch",
        "tqdm>=4.64.1",
        "matplotlib>=3.5.3",
        "seaborn>=0.12.0",
        "numpy>=1.23.3",
        "diffusers>=0.7.2",
        "scipy>=1.9.3",
        "setuptools~=65.5.0",
        "pillow>=9.2.0",
        "mpi4py>=3.1.4",
        "torchvision>=0.14.0"
    ],
)
