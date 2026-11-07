"""An in-thread fake of `aismith-engine serve`, the engine sidecar's HTTP runner (contract v1)."""
import hashlib
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

GLB = b'glTF' + bytes(124)
JOB = r'(?P<job>[A-Za-z0-9_-]{1,64})'


class FakeRunner:
    """Implements health, inspect, PUT input, POST run, GET output, cancel and DELETE.

    `control['run']` steers a run: `status`, `error`, `sleep` (seconds, cut short by
    cancel or DELETE), `effective_inputs`, `output` (fields merged into the result's
    output), `result` (merged into the result), `http_status`, `body` (served bytes)
    and `x_sha256` (served header).
    """

    def __init__(self, inspect_report: dict):
        self.control: dict = {'inspect': inspect_report, 'run': {}}
        self.calls: list[tuple[str, str]] = []
        self.inputs: dict[str, dict[str, bytes]] = {}
        self.runs: list[dict] = []
        self._stop: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        runner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def _send(self, status, body=b'', content_type='application/json', headers=None):
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(body)

            def _json(self, status, value):
                self._send(status, json.dumps(value).encode())

            def _route(self, method):
                url = urlsplit(self.path)
                with runner._lock:
                    runner.calls.append((method, self.path))
                return url.path, parse_qs(url.query)

            def do_GET(self):
                path, query = self._route('GET')
                if path == '/health':
                    return self._json(200, {'ok': True, 'busy': False, 'schema_version': 1})
                if path == '/inspect':
                    time.sleep(runner.control.get('inspect_sleep', 0))
                    return self._json(runner.control.get('inspect_status', 200), runner.control['inspect'])
                match = re.fullmatch(rf'/jobs/{JOB}/output', path)
                if match:
                    run = runner.control['run']
                    body = run.get('body', GLB)
                    return self._send(200, body, 'model/gltf-binary', {
                        'X-Sha256': run.get('x_sha256', hashlib.sha256(GLB).hexdigest())})
                return self._json(404, {'error': 'not found'})

            def do_PUT(self):
                path, _ = self._route('PUT')
                match = re.fullmatch(rf'/jobs/{JOB}/inputs/(?P<name>front|back|left|right|mesh)', path)
                if not match:
                    return self._json(400, {'error': 'bad name or id'})
                if self.headers.get('Transfer-Encoding') or self.headers.get('Content-Length') is None:
                    return self._json(411, {'error': 'Content-Length is required'})
                body = self.rfile.read(int(self.headers['Content-Length']))
                runner.inputs.setdefault(match['job'], {})[match['name']] = body
                return self._send(204)

            def do_POST(self):
                path, _ = self._route('POST')
                match = re.fullmatch(rf'/jobs/{JOB}/(?P<action>run|cancel)', path)
                if not match:
                    return self._json(404, {'error': 'not found'})
                job = match['job']
                if match['action'] == 'cancel':
                    runner._stop.setdefault(job, threading.Event()).set()
                    return self._send(202)
                stop = runner._stop[job] = threading.Event()
                length = int(self.headers.get('Content-Length') or 0)
                if length > 64 * 1024:
                    return self._json(413, {'error': 'run body too large'})
                spec = json.loads(self.rfile.read(length))
                deadline = spec.get('deadline_seconds')
                if isinstance(deadline, bool) or not isinstance(deadline, int) or not 0 < deadline <= 86400:
                    return self._json(400, {'error': 'deadline_seconds must be an integer from 1 to 86400'})
                runner.runs.append(spec)
                run = runner.control['run']
                if run.get('http_status'):
                    return self._json(run['http_status'], {'error': 'busy'})
                stop.wait(run.get('sleep', 0))
                if stop.is_set():
                    # Runner-made failures come back as result-schema JSON.
                    return self._json(200, {'stage': spec['stage'], 'status': 'failed', 'schema_version': 1,
                                            'error': 'cancelled', 'engine': runner.control['inspect']['engine'],
                                            'diagnostics': {'runner': {'returncode': -15, 'log_tail': ''}}})
                status = run.get('status', 'ok')
                result = {'stage': spec['stage'], 'status': status, 'schema_version': 1, 'output': None,
                          'gate': None, 'effective_inputs': {}, 'measured': {},
                          'engine': runner.control['inspect']['engine'],
                          'weights': {'ok': True, 'pinned': True, 'files': []}, 'diagnostics': {},
                          'timings_s': {'total': 1.5}, 'error': run.get('error')}
                if status == 'ok':
                    result['output'] = {'url': f'/jobs/{job}/output', 'content_type': 'model/gltf-binary',
                                        'sha256': hashlib.sha256(GLB).hexdigest(), 'bytes': len(GLB),
                                        **run.get('output', {})}
                    result['effective_inputs'] = run.get(
                        'effective_inputs', {**spec['settings'], 'views': sorted(spec['inputs'])})
                    result['measured'] = {'triangles': 1234, 'watertight': True}
                result.update(run.get('result', {}))
                return self._json(200, result)

            def do_DELETE(self):
                path, _ = self._route('DELETE')
                match = re.fullmatch(rf'/jobs/{JOB}', path)
                if not match:
                    return self._json(400, {'error': 'bad id'})
                runner._stop.setdefault(match['job'], threading.Event()).set()
                runner.inputs.pop(match['job'], None)
                return self._send(204)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f'http://127.0.0.1:{self.server.server_address[1]}'

    def methods(self, job: str | None = None) -> list[str]:
        """`METHOD /path` for each call, optionally only one job's."""
        return [f'{method} {path}' for method, path in self.calls
                if job is None or f'/jobs/{job}/' in path + '/']

    def close(self) -> None:
        for stop in self._stop.values():
            stop.set()
        self.server.shutdown()
        self.server.server_close()
