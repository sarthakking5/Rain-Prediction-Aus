from setuptools import setup,find_packages
with open("requirements.txt") as f:
    requirements=f.read().splitlines()

    setup(
        name="rain_prediction",
        version="0.1",
        author="Sarthak",
        packages=find_packages(),
        install_requires=requirements,
    ) 