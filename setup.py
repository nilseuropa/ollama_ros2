from setuptools import setup

package_name = 'ollama_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/ollama_node.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nils',
    maintainer_email='nils@example.com',
    description='ROS 2 node bridging std_msgs/String to Ollama local LLM via HTTP.',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ollama_node = ollama_ros2.ollama_node:main',
        ],
    },
)

