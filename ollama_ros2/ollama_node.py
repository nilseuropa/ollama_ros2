import json
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String

try:
    # Prefer stdlib to avoid external deps
    import urllib.request as _urllib_request
    import urllib.error as _urllib_error
except Exception:  # pragma: no cover
    _urllib_request = None
    _urllib_error = None


class OllamaBridge(Node):
    def __init__(self) -> None:
        super().__init__('ollama_bridge')

        # Parameters
        self.declare_parameter('api_url', 'http://127.0.0.1:11434')
        self.declare_parameter('endpoint', '/api/generate')  # or '/api/chat'
        self.declare_parameter('model', 'llama3.2')
        self.declare_parameter('stream', False)
        self.declare_parameter('timeout_sec', 120.0)
        self.declare_parameter('temperature', 0.7)
        self.declare_parameter('top_p', 0.9)
        self.declare_parameter('system_prompt', '')
        self.declare_parameter('input_topic', 'input')
        self.declare_parameter('output_topic', 'output')

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        # Pub/Sub
        self.publisher_ = self.create_publisher(String, output_topic, 10)
        self.subscription = self.create_subscription(
            String, input_topic, self._on_input, 10
        )

        # Resolve and possibly adjust the model parameter based on Ollama availability
        try:
            configured_model = self.get_parameter('model').get_parameter_value().string_value
            effective_model = self._resolve_model(configured_model)
            self._model = effective_model
            if effective_model != configured_model:
                # Update the parameter so external queries reflect the effective model
                self.set_parameters([Parameter('model', value=effective_model)])
                self.get_logger().warn(
                    f"Requested model '{configured_model}' not available; using '{effective_model}'."
                )
        except Exception as e:
            # If resolution fails (e.g., Ollama unavailable), proceed with configured model
            self._model = self.get_parameter('model').get_parameter_value().string_value
            self.get_logger().warn(
                f"Could not resolve models from Ollama: {e}. Proceeding with '{self._model}'."
            )

        self.get_logger().info(
            f"OllamaBridge ready: subscribing '{input_topic}', publishing '{output_topic}'."
        )
        # Log effective key parameters to avoid confusion when launch files override defaults
        self.get_logger().info(
            f"Configured endpoint='{self.get_parameter('endpoint').value}', model='{self._model}'."
        )

    def _on_input(self, msg: String) -> None:
        prompt = msg.data.strip()
        if not prompt:
            self.get_logger().warn('Received empty prompt; skipping.')
            return

        try:
            response_text = self._query_ollama(prompt)
        except Exception as exc:  # Catch and publish error as output
            err = f"[ollama_ros2 error] {type(exc).__name__}: {exc}"
            self.get_logger().error(err)
            out = String()
            out.data = err
            self.publisher_.publish(out)
            return

        out = String()
        out.data = response_text
        self.publisher_.publish(out)

    def _query_ollama(self, prompt: str) -> str:
        api_url = self.get_parameter('api_url').get_parameter_value().string_value.rstrip('/')
        endpoint = self.get_parameter('endpoint').get_parameter_value().string_value
        model = getattr(self, '_model', None) or self.get_parameter('model').get_parameter_value().string_value
        stream = self.get_parameter('stream').get_parameter_value().bool_value
        timeout_sec = self.get_parameter('timeout_sec').get_parameter_value().double_value
        temperature = self.get_parameter('temperature').get_parameter_value().double_value
        top_p = self.get_parameter('top_p').get_parameter_value().double_value
        system_prompt = self.get_parameter('system_prompt').get_parameter_value().string_value

        url = f"{api_url}{endpoint}"

        if endpoint == '/api/chat':
            # Chat API expects messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "options": {"temperature": temperature, "top_p": top_p},
            }
        else:
            # Generate API
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            payload = {
                "model": model,
                "prompt": full_prompt,
                "stream": stream,
                "options": {"temperature": temperature, "top_p": top_p},
            }

        data = json.dumps(payload).encode('utf-8')
        req = _urllib_request.Request(url, data=data, headers={'Content-Type': 'application/json'})

        start = time.time()
        try:
            with _urllib_request.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read()
        except _urllib_error.HTTPError as e:
            body = e.read().decode('utf-8', errors='ignore') if hasattr(e, 'read') else ''
            raise RuntimeError(f"HTTP {e.code} from Ollama: {body}") from e
        except _urllib_error.URLError as e:
            raise RuntimeError(f"Failed to reach Ollama at {url}: {e}") from e

        elapsed = (time.time() - start) * 1000.0
        self.get_logger().info(f"Ollama responded in {elapsed:.0f} ms")

        txt = raw.decode('utf-8', errors='ignore')
        # If streaming was disabled, expect a single JSON
        try:
            obj = json.loads(txt)
            if 'response' in obj:
                return obj.get('response', '')
            # Chat API returns message in message.content
            message = obj.get('message', {})
            if isinstance(message, dict):
                return message.get('content', '')
            return ''
        except json.JSONDecodeError:
            # If stream=True, it might be NDJSON; concatenate 'response' chunks
            responses = []
            for line in txt.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    part = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'response' in part:
                    responses.append(part['response'])
                elif 'message' in part and isinstance(part['message'], dict):
                    responses.append(part['message'].get('content', ''))
        return ''.join(responses)

    def _resolve_model(self, requested: str) -> str:
        """Return an available model name.

        If the requested model is not available, choose the first available.
        Tries to match requested name without tag to a tagged model (e.g., 'llama3.2' -> 'llama3.2:latest').
        Raises RuntimeError if no models are available from Ollama.
        """
        api_url = self.get_parameter('api_url').get_parameter_value().string_value.rstrip('/')
        tags_url = f"{api_url}/api/tags"

        req = _urllib_request.Request(tags_url, headers={'Accept': 'application/json'})
        try:
            with _urllib_request.urlopen(req, timeout=10.0) as resp:
                raw = resp.read()
        except _urllib_error.HTTPError as e:
            body = e.read().decode('utf-8', errors='ignore') if hasattr(e, 'read') else ''
            raise RuntimeError(f"HTTP {e.code} from Ollama tags: {body}") from e
        except _urllib_error.URLError as e:
            raise RuntimeError(f"Failed to reach Ollama at {tags_url}: {e}") from e

        data = json.loads(raw.decode('utf-8', errors='ignore'))
        models = data.get('models', [])
        names = [m.get('name') for m in models if isinstance(m, dict) and m.get('name')]

        if not names:
            raise RuntimeError('No models available from Ollama.')

        # Exact match
        if requested in names:
            return requested

        # If user omitted tag, try to find a tagged variant like ':latest'
        if ':' not in requested:
            for n in names:
                if n.startswith(requested + ':'):
                    return n

        # Fallback to first available
        return names[0]


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = OllamaBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
