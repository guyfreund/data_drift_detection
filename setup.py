from setuptools import setup, find_packages

VERSION = '1.0.0'
DESCRIPTION = 'Data Drift Detection - MLOps project, Reichman University 2022\n'
LONG_DESCRIPTION = 'Implementation of a Machine Learning Operations pipeline, which consists of: model training, ' \
                   'data drift detection & data generation to generate data drift.\n' \
                   'Databases being used are: \n' \
                   '- https://archive.ics.uci.edu/ml/datasets/bank+marketing\n' \
                   '- https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)\n'

setup(
    name="data_drift_detection",
    version=VERSION,
    author="Guy Freund, Danielle Ben-Bashat, Elad Prager",
    author_email="guyfreund@gmail.com",
    description=DESCRIPTION,
    # license="MIT",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        "certifi==2021.10.8",
        "cycler==0.11.0",
        "cython==0.29.26",
        "distlib==0.3.4",
        "easydict==1.9",
        "filelock==3.4.2",
        "fonttools==4.28.5",
        "imbalanced-learn==0.8.1",
        "imblearn==0.0",
        "joblib==1.1.0",
        "kiwisolver==1.3.2",
        "matplotlib==3.5.1",
        "numpy==1.21.5",
        "packaging==21.3",
        "pandas==1.3.5",
        "pillow==8.4.0",
        "pipenv==2021.11.23",
        "platformdirs==2.4.1",
        "pyparsing==3.0.6",
        "python-dateutil==2.8.2",
        "pytz==2021.3",
        "pyyaml==6.0",
        "ruamel.yaml.clib==0.2.6",
        "ruamel.yaml==0.17.19",
        "scikit-learn==1.0.2",
        "scipy==1.7.3",
        "seaborn==0.11.2",
        "setuptools==60.2.0",
        "six==1.16.0",
        "sortedcontainers==2.4.0",
        "threadpoolctl==3.0.0",
        "virtualenv-clone==0.5.7",
        "virtualenv==20.12.0",
        "xgboost==1.5.1"
    ],
    python_requires=">=3.8.12",
    url="https://github.com/guyfreund/data_drift_detector/",
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "data_drift_detector=pipeline_manager.__main__:main",
        ]
    },
)
