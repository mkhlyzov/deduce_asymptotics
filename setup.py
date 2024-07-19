from setuptools import setup, find_packages

setup(
    name="deduce_asymptotics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy"
    ],
    author="Michael Khlyzov",
    author_email="s-man911@ukr.net",
    description="A package to deduce the time complexity of functions.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/mkhlyzov/deduce_asymptotics",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
