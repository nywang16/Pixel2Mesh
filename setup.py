from setuptools import setup
from setuptools import find_packages

setup(name='pixel2mesh',
	version='1.0',
	description='Implementation of Pixel2Mesh in TensorFlow',
	author='Nanyang Wang',
	author_email='nywang16@163.com',
	url='http://bigvid.fudan.edu.cn/pixel2mesh',
	download_url='https://github.com/nywang16/Pixel2Mesh',
	license='MIT',
	install_requires=['numpy', 'tflearn', 'opencv-python'],
	package_data={'pixel2mesh': ['README.md']},
	packages=find_packages())
