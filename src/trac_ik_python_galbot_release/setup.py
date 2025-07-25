from setuptools import setup, find_packages

setup(
    name="trac_ik_python_galbot_release",
    version="0.0.1",
    description="TRAC-IK Python binding with all dependencies included",
    author="Chenyu Cao, Yuxin Yang",
    author_email="caochenyu@galbot.com, yangyuxin@galbot.com",
    packages=["trac_ik_python"],
    package_dir={"trac_ik_python": "src/trac_ik_python"},
    package_data={
        "trac_ik_python": ["*.so", "*.so.*", "*.py"],
    },
    include_package_data=True,
    python_requires=">=3.10,<3.11",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license_files = ("LICENSE.txt",),
)