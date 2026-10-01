from setuptools import setup
from setuptools.dist import Distribution


class PlatformBinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(distclass=PlatformBinaryDistribution)
