import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

REQUIRES = [
    "pandas",
    "numpy",
    "scipy",
    "torch",
    "scikit-learn",
    "matplotlib",
    "dython",
    "tqdm",
    "pytest"
]

setuptools.setup(
    name="relgen",
    version="0.1.1",
    author="ruc-datalab",
    author_email="ltyzzz@ruc.edu.cn",
    description="A tool for relational data generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ruc-datalab/RelGen",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    license="Apache-2.0",
    install_requires=REQUIRES,
)
