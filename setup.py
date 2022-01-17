from pathlib import Path

from setuptools import setup, find_packages

long_description = Path("README.md").read_text("utf-8")

try:
    from wordle import __email__, __author__, __version__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ""
    __version__ = "0.0.0"

setup(
    name="wordle",
    version=__version__,
    description="Word guesses in Wordle",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ilibarra/wordle_solver",
    author=__author__,
    author_email=__email__,
    license="GNU",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[l.strip() for l in Path("requirements.txt").read_text("utf-8").splitlines()],
    extras_require=dict(
        dev=["pre-commit>=2.7.1"],
        test=["tox>=3.20.1"],
        docs=[
            l.strip()
            # for l in (Path("docs") / "requirements.txt").read_text("utf-8").splitlines()
            # if not l.startswith("-r")
            for l in (Path(".") / "requirements.txt").read_text("utf-8").splitlines()
            # if not l.startswith("-r")

        ],
    ),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Environment :: Console",
        "Framework :: Jupyter",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
    ],
)
