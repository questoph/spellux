from setuptools import setup

setup(name='spellux',
      version='0.1.1',
      description='Automatic text normalization for Luxembourgish',
      url='https://github.com/questoph/spellux',
      author='Christoph Purschke',
      author_email='christoph@purschke.info',
      license='MIT',
      packages=['spellux'],
      zip_safe=False,
      include_package_data=True,
      python_requires=">=3.6")
