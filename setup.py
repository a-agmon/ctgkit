from setuptools import setup, find_packages

setup(
    name="ctgkit",
    version="0.1.3",
    description="Fetal heart-rate epoch analysis (decision support only)",
    packages=find_packages(),
    install_requires=["numpy", "scipy"],
    extras_require={"plot": ["matplotlib"], "io": ["pandas"]},
    python_requires=">=3.9",
)
