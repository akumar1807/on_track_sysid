from setuptools import setup

package_name = 'on_track_sysid'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ayush',
    maintainer_email='ayush18.kumar@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sim_collect = on_track_sysid.collect_data_for_sys_id:main',
            'with_data_sys_id = on_track_sysid.with_data_sys_id:main',
            'jetson_sys_id = on_track_sysid.jetson_sys_id:main',
            'ontrack = on_track_sysid.on_track_jetson:main',
            'jetson_collect = on_track_sysid.collect_data_jetson:main'
        ],
    },
)
