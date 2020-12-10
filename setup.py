import setuptools

setuptools.setup(
    name='dynamol',
    version='1.0.0',
    author="Vagner Bessa",
    author_email="bessavagner@gmail.com",
    url="https://github.com/bessavagner",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'h5py'
    ]
)
