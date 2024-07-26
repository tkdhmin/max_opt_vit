import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MaxOptVit",
    version="1.0.0",
    author="Donghyun Min",
    author_email="mdh38112@sogang.ac.kr",
    description="A collection of python-based deep learning model implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tkdhmin/max_opt_vit.git",
    project_urls={
        "Bug Tracker": "https://github.com/tkdhmin/max_opt_vit/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "models"},
    packages=setuptools.find_packages(where="models"),
    python_requires=">= 3.8",
    install_requires=[
        # external dependencies
        "numpy               == 1.22.3",
        "sympy               >= 1.11",
        "mock                == 4.0.3",
        "nbformat            == 5.6.1",
        "importlib-resources >= 5.9.0",
        "pandas              == 1.5.1",
    ],
    extras_require={
        "dev": ["pytest>=6.0"],
    },
)
