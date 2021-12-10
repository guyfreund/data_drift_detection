from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Data Drift Detection - MLOps project, Reichman University 2022\n'
LONG_DESCRIPTION = 'Implementation of a Machine Learning Operations pipeline, which consists of: model training, ' \
                   'data drift detection & data slicing with regards to fairness.\n' \
                   'Databases being used are: \n' \
                   '- https://archive.ics.uci.edu/ml/datasets/bank+marketing\n' \
                   '- https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)\n'

setup(
    name="data_drift_detector",
    version=VERSION,
    author="Guy Freund, Danielle Ben-Bashat, Elad Prager",
    author_email="guyfreund@gmail.com",
    description=DESCRIPTION,
    license="MIT",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.10",
    url="https://github.com/guyfreund/data_drift_detctor/",
    classifiers=[
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License"
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)