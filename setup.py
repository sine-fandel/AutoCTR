from setuptools import setup
import setuptools

with open ("README.md", "r") as fh :
	long_description = fh.read ()

setup (
	name="autoctr",
	version="1.0.0",
	author="Zhengxin Fang",
	author_email="358746595@qq.com",
	description="Automated Machine Learning for CTR",
	long_description=long_description,
	long_description_content_type="text/markdown",
	packages=setuptools.find_packages(),
)