from setuptools import setup, find_packages

setup(
    name="shine_lab_code",
    version="0.1.0",
    author="Florian Burger",
    author_email="fburger20@gmx.de",
    description="A Python package for functional brain signal analysis (ICG, LFA, Spectral, etc.)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Flo2306/Shine_Lab_Combined_Code",  # Replace with your GitHub repo
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
