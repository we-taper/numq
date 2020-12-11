from setuptools import find_packages, setup
import versioneer

with open("requirements.txt", "r") as f:
    requirements = f.readlines()
    requirements = [_r.strip() for _r in requirements]
    requirements = [_ for _ in requirements if (not _.startswith("#")) and (_ != "")]

setup(
    name="numq",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(
        include=("src/numq.*")
    ),
    # metadata to display on PyPI
    author="Hongxiang Chen",
    author_email="h.chen.17@ucl.ac.uk",
    description="Numerical tools for quantum.",
    license="GNU GPLv3",
    install_requires=requirements,
)
