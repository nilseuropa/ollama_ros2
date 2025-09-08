Ollama ROS 2 Bridge
===================

A simple ROS 2 Python node that subscribes to a `std_msgs/String` prompt and publishes the model response from a local Ollama server.

Topics
------
- input: `std_msgs/String` with the prompt
- output: `std_msgs/String` with the response or error text

Parameters
----------
- api_url (string): Base URL of Ollama (default `http://127.0.0.1:11434`)
- endpoint (string): `/api/generate` or `/api/chat` (default `/api/generate`)
- model (string): Model name (default `llama3.2`)
- stream (bool): If true, concatenates streaming chunks (default false)
- timeout_sec (double): HTTP timeout seconds (default 120.0)
- temperature (double): Sampling temperature (default 0.7)
- top_p (double): Top-p sampling (default 0.9)
- system_prompt (string): Optional system prompt (default empty)
- input_topic (string): Input topic name (default `input`)
- output_topic (string): Output topic name (default `output`)
- enable_tts (bool): If true, sends the response to a TTS action (default false)
- tts_action_name (string): Action name for TTS (default `/say`)

Build & Run
-----------
1. Source your ROS 2 setup and build the workspace:
   - `source /opt/ros/$ROS_DISTRO/setup.bash`
   - `colcon build`
   - `source install/setup.bash`

2. Ensure Ollama is running locally and the chosen `model` exists (e.g., `ollama run llama3.2`).

3. Launch the node:
   - `ros2 launch ollama_ros2 ollama_node.launch.py`

4. Interact via topics:
   - `ros2 topic pub /input std_msgs/String "data: 'Write a haiku about robots.'" -1`
   - `ros2 topic echo /output`
   - Enable TTS: `ros2 launch ollama_ros2 ollama_node.launch.py enable_tts:=true`
     - or: `ros2 run ollama_ros2 ollama_node --ros-args -p enable_tts:=true -p tts_action_name:=/say`

Notes
-----
- The node uses Python stdlib `urllib` to avoid extra dependencies.
- Switch to `/api/chat` if you prefer chat-style prompts and roles.
- On startup, the node queries Ollama for available models (`/api/tags`). If the configured `model` is not found, it selects the first available model (and updates the `model` parameter accordingly). It also attempts to match bare names to tagged variants (e.g., `llama3.2` -> `llama3.2:latest`).
