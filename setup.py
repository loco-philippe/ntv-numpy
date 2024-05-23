# -*- coding: utf-8 -*-
"""
@author: philippe@loco-labs.io
"""

import pathlib
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="ntv_numpy",
    version="0.2.2",
    description="NTV-NumPy : A multidimensional semantic, compact and reversible format for interoperability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/loco-philippe/ntv-numpy/blob/main/README.md",
    author="Philippe Thomy",
    author_email="philippe@loco-labs.io",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="numpy, JSON-NTV, semantic JSON, development, environmental data, multidimensional",
    packages=find_packages(include=["ntv_numpy", "ntv_numpy.*"]),
    package_data={"ntv_numpy": ["*.ini"]},
    python_requires=">=3.10, <4",
    install_requires=["json_ntv", "numpy", "shapely"],
)
