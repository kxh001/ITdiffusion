from setuptools import setup

setup(
    name="diffusion",
    py_modules=["utilsitd"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
