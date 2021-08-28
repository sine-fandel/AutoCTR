from setuptools import setup

with open ("README.md", "r") as fh :
	long_description = fh.read ()

setup (
	name="AutoCTR",
	version="0.0.1",
	author="Zhengxin Fang",
	author_email="358746595@qq.com",
	description="Automated Machine Learning for CTR",
	long_description=long_description,
	long_description_content_type="text/markdown",
	packages=setuptools.find_packages(),
)