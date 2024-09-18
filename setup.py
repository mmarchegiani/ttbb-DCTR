from setuptools import find_packages, setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ttbb_dctr",
    packages=find_packages(),
    version="0.1.0",
    description="Repository for development of the ttbb MC reweighting in the ttH(bb) analysis.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Matteo Marchegiani",
    author_email="matteo.marchegiani@cern.ch",
    url="https://github.com/mmarchegiani/ttbb-DCTR",
    license="BSD-3-Clause",
    install_requires=[
        'vector==1.4.1',
        'omegaconf',
        'scikit-learn',
        'htcondor',
        'pytorch_lightning',
        'seaborn'
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Environment :: Console',
        'Environment :: GPU :: NVIDIA CUDA :: 11.7',
        'Environment :: GPU :: NVIDIA CUDA :: 11.8',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
        'Typing :: Typed'
    ],
)
