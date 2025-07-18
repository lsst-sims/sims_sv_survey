from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sv_survey")
except PackageNotFoundError:
    # package is not installed
    pass
