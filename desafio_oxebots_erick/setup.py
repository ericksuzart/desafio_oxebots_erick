import os
from glob import glob

from setuptools import setup

package_name = 'desafio_oxebots_erick'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Erick Suzart',
    maintainer_email='erick.suzart@ufba.br',
    description='Desafio OxeBots - Hands On Wall Follower',
    license='GPL-3.0-only',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_controller = desafio_oxebots_erick.robot_controller:main',
        ],
    },
)
