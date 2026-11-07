"""The http transport against the real `aismith-engine serve`, driving the fake engine CLI.

Opt-in: set AISMITH_ENGINE_SERVE_ROOT to an AISmith-3D checkout that has
`backend/engine/serve.py`. The runner is AISmith's own code; only the engine
CLI it launches is this suite's fake, so no GPU or weights are needed.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from outbound_gpu_worker_pool.engine import (
    AISmithEnginePlugin, EngineDrift, EngineFailed, engine_versions, prepare_engine_environment,
)
from outbound_gpu_worker_pool.engine_recipes import MESH3D
from outbound_gpu_worker_pool.plugins import PluginRequestRejected
from test_engine_plugin import FakeEngine, _drift, _gone, context, inspect_report, lease

ROOT = os.environ.get('AISMITH_ENGINE_SERVE_ROOT', '')
pytestmark = pytest.mark.skipif(
    not (ROOT and (Path(ROOT) / 'backend' / 'engine' / 'serve.py').is_file()),
    reason='set AISMITH_ENGINE_SERVE_ROOT to an AISmith-3D checkout with backend/engine/serve.py')

SERVE = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from backend.engine.serve import make_server
server = make_server('127.0.0.1', 0, work_root=Path(sys.argv[2]), quiet=True,
                     engine_command=[sys.executable, sys.argv[3]], kill_grace=1)
print(server.server_address[1], flush=True)
server.serve_forever()
'''


@pytest.fixture
def engine(tmp_path):
    directory = tmp_path / 'engine'
    directory.mkdir()
    return FakeEngine(directory)


@pytest.fixture
def runner_url(engine, tmp_path):
    process = subprocess.Popen(
        [sys.executable, '-c', SERVE, ROOT, str(tmp_path / 'runner-work'), str(engine.directory / 'fake_engine.py')],
        stdout=subprocess.PIPE, cwd=ROOT)
    try:
        port = int(process.stdout.readline())
        yield f'http://127.0.0.1:{port}'
    finally:
        process.terminate()
        process.wait(10)


@pytest.fixture
def plugin(runner_url):
    return AISmithEnginePlugin(MESH3D, engine_versions(MESH3D, inspect_report()), engine_url=runner_url)


async def test_enrolment_and_a_mesh_job_through_the_real_runner(plugin, runner_url, engine, tmp_path):
    settings = await prepare_engine_environment(MESH3D, {'OGWP_ENGINE_URL': runner_url})
    assert settings['OGWP_ENGINE_URL'] == runner_url
    output = await plugin.execute(context(tmp_path), plugin.validate(lease()))
    assert output.path.read_bytes().startswith(b'glTF')
    assert output.diagnostics['effective_inputs'] == {'resolution': 1024, 'seed': 7}
    assert [call[:2] for call in engine.calls()] == [['inspect', '--full'], ['inspect'], ['run', 'mesh']]
    # DELETE removed the job's work directory on the runner.
    assert not any((tmp_path / 'runner-work').glob('*job-3d*'))


async def test_a_rejected_spec_through_the_real_runner(plugin, engine, tmp_path):
    engine.set(status='rejected', error="input 'front' is not an image")
    with pytest.raises(PluginRequestRejected, match='not an image'):
        await plugin.execute(context(tmp_path), plugin.validate(lease()))


async def test_drift_through_the_real_runner(plugin, engine, tmp_path):
    engine.control['inspect'] = _drift(inspect_report(), 'commit')
    engine.save()
    with pytest.raises(EngineDrift):
        await plugin.execute(context(tmp_path), plugin.validate(lease()))


async def test_the_deadline_cancels_the_real_runner_job(plugin, engine, tmp_path):
    engine.set(child=True, sleep=300)
    started = time.monotonic()
    # Whichever enforces the deadline first: the runner ("deadline exceeded",
    # killing the CLI's group) or the client (POST cancel); both fail the job.
    with pytest.raises(EngineFailed, match='deadline'):
        await plugin.execute(context(tmp_path, deadline_seconds=8), plugin.validate(lease()))
    assert time.monotonic() - started < 40
    assert _gone(int((engine.directory / 'child.pid').read_text()))
