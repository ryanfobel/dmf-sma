import sys

from paver.setuputils import install_distutils_tasks
from paver.easy import task, needs, options

sys.path.insert(0, '.')
import version


properties = dict(
      package_name='dmf_sma',
      version=version.getVersion(),
      url='http://github.com/wheeler-microfluidics/dmf-sma.git',
      short_description='Code for simulating, modeling, and analysing Digital '
                        'Microfluidics force and velocity data.',
      long_description='',
      category='Analysis',
      author='Ryan Fobel',
      author_email='ryan@fobel.net')


install_distutils_tasks()

options(
    LIB_PROPERTIES=properties,
    setup=dict(name=properties['package_name'].replace('_', '-'),
               description='\n'.join([properties['short_description'],
                                      properties['long_description']]),
               author_email=properties['author_email'],
               author=properties['author'],
               url=properties['url'],
               version=properties['version'],
               install_requires=['numpy', 'pandas', 'path-helpers>=0.2'],
               # Install data listed in `MANIFEST.in`
               include_package_data=True,
               license='GPLv2',
               packages=[properties['package_name']]))


@task
@needs('generate_setup', 'minilib', 'setuptools.command.sdist')
def sdist():
    """Override sdist to make sure that our setup.py is generated."""
    pass
