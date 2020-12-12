import os

from setuptools import find_packages, setup

import versioneer

with open("requirements.txt", "r") as f:
    requirements = f.readlines()
    requirements = [_r.strip() for _r in requirements]
    requirements = [_ for _ in requirements if (not _.startswith("#")) and (_ != "")]
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'README.md')
with open(README_PATH) as readme_file:
    README = readme_file.read()
setup(
    name="numq",
    version=versioneer.get_version(),
    description="Numerical tools for quantum experiments.",
    long_description=README,
    long_description_content_type='text/markdown',
    url="https://github.com/we-taper/numq",
    project_urls={
        "Source Code": "https://github.com/we-taper/numq",
    },
    author="Taper",
    author_email="h.chen.17@ucl.ac.uk",
    cmdclass=versioneer.get_cmdclass(),
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.6, <4',
    license="GNU GPLv3",
    install_requires=requirements,
)
