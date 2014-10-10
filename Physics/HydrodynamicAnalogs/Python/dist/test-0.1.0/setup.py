from distutils.core import setup

setup(
    name='test',
    version='0.1.0',
    author='J. Random Hacker',
    author_email='jrh@example.com',
    packages=['Visibility'],
    scripts=['bin/Visibility.py'],
    url='http://pypi.python.org/pypi/TowelStuff/',
    license='gpl-2.0.txt',
    description='Useful towel-related stuff.',
    long_description=open('README.txt').read(),
    install_requires=[
        "Django >= 1.1.1",
        "caldav == 0.1.4",
    ],
)