from setuptools import setup, find_packages

setup(
    name='pytorchutils',
    version='0.0.1',
    keywords='pytorch utils',
    description='a library for using pytorch easier',
    license='MIT License',
    url='https://github.com/dong100136/pytorchutils',
    author='Stone Ye',
    author_email='dong100136@outlook.com',
    packages=find_packages(),
    include_package_data=True,
    platforms='any',
    install_requires=[torch,torchvision],
)
