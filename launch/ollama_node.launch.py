from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ollama_ros2',
            executable='ollama_node',
            name='ollama_bridge',
            output='screen',
            parameters=[
                {'api_url': 'http://127.0.0.1:11434'},
                {'endpoint': '/api/generate'},  # or '/api/chat'
                {'model': 'krumpli'},
                {'stream': False},
                {'timeout_sec': 120.0},
                {'temperature': 0.7},
                {'top_p': 0.9},
                {'system_prompt': ''},
                {'input_topic': 'input'},
                {'output_topic': 'output'},
                {'enable_tts': True},
                {'tts_action_name': '/say'},
            ],
        )
    ])
