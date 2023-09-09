from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="caidin",
    version="0.0.1",
    author="Emmanuel Olisah",
    author_email="manuelolisah@gmail.com",
    description="An open-source recommendation engine library in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/netesy/caidin",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # List your required packages here
    ],
    python_requires=">=3.6",
)
