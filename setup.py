import setuptools
from os import path

PARC_DIR = path.abspath(path.dirname(__file__))

with open(path.join(PARC_DIR, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="parc",
    version="0.40",
    packages=["parc"],
    license="MIT",
    author_email="shobana.venkat88@gmail.com",
    url="https://github.com/ShobiStassen/PARC",
    setup_requires=["numpy", "pybind11"],
    install_requires=[
        line.strip() for line in open("requirements.txt")
    ],
    extras_require={
        "dev": ["pytest", "scikit-learn"]
    },
    long_description=long_description,
    long_description_content_type="text/markdown"
)
