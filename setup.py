
from setuptools import setup


setup(name="FunGCN",
      packages=["fungcn"],
      install_requires=["numpy>=1.26.2", "scikit-learn>=1.1.2", "tqdm>=4.66.1", "scipy>=1.11.4", "matplotlib>=3.8.2",
                        "pandas>=2.1.3", "torch>=2.1.1", "torch_geometric>=2.4.0", "xarray>=2023.11.0"])
