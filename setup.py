from setuptools import setup

with open("README.md", 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(name='ls_python',
    version='0.1',
    description='Level set library written in Python',
    url='https://github.com/rahulverma88/ls_python',
    author='Rahul Verma',
    author_email='rahulverma88@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',  
    license='MIT',
    packages=['ls_python'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
    zip_safe=False
)
