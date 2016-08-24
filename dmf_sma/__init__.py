import pkg_resources
from path_helpers import path

# get the python package name based on the name of the parent directory
PACKAGE_NAME = pkg_resources.get_distribution(str(path(__file__).parent.name)).project_name
