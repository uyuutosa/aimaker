from setuptools import setup, find_packages

requires = ["requests>=2.14.2"]


setup(
    name='aimaker',
    version='0.1.4',
    description='an end-to-end AI training and hosting tool',
    url='https://github.com/uyuutosa/aimaker',
    author='uyuutosa',
    author_email='sayu819@gmail.com',
    keywords='AI',
    packages=find_packages(),
    #install_requires=["scikit-learn"],
    classifiers=[
                'Programming Language :: Python :: 3.6',
            ],
)
