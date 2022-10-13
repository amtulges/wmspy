from setuptools import setup

setup(name='wmspy',
      version='0.0',
      author='Andrew Tulgestke',
      author_email='amtulges@gmail.com',
      packages=[
          'wmspy',
          'wmspy.test',
          ],
      scripts=[
          ],
      package_data={'wmspy':['data/*.yaml',
                             'data/*.yml',
                             'data/*.txt',
                             ]},
      url='',
      license='LICENSE.txt',
      description='A package to aid in analyzing Wavelength Modulation Spectroscopy (WMS) measurements.',
      long_description=open('README.txt').read(),
      install_requires=[
          'dataclasses',
          'matplotlib',
          'numpy',
          'pandas',
          'pathlib',
          'scipy',
          ],
      )
