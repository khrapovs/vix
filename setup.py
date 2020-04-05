from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='vix',
    version='',
    packages=['vix'],
    url='https://github.com/khrapovs/vix',
    license='MIT',
    author='Stanislav Khrapov',
    author_email='khrapovs@gmail.com',
    description='Compute VIX and its derivatives',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
    install_requires=open('requirements.txt').read()
)
