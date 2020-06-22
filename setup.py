from io import open
from os import path

from setuptools import setup  # find_packages

here = path.abspath(path.dirname(__file__))

# get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


# get reqs
def requirements():
    list_requirements = []
    with open('requirements.txt') as f:
        for line in f:
            list_requirements.append(line.rstrip())
    return list_requirements


setup(
      name='auxilearn',
      version='1.0.0',  # Required
      description='AuxiLearn',  # Optional
      long_description='AuxiLearn - Auxiliary Learning by Implicit Differentiation',  # Optional
      long_description_content_type='text/markdown',  # Optional (see note above)
      url='',  # Optional
      author='',  # Optional
      author_email='',  # Optional
      packages=['auxilearn'],
      # packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
      python_requires='>=3.5',
      install_requires=requirements()  # Optional
)
