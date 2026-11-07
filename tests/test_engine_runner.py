"""The plugin's `http` transport against an in-thread fake of `aismith-engine serve`."""
import asyncio
import hashlib
import time

import pytest

from outbound_gpu_worker_pool.engine import (
    AISmithEnginePlugin, EngineCancelled, EngineDrift, EngineFailed, EngineTimedOut, engine_versions,
    prepare_engine_environment, runner_url,
)
from outbound_gpu_worker_pool.engine_recipes import MESH3D
from outbound_gpu_worker_pool.plugins import PluginRequestRejected
from fake_engine_runner import GLB, FakeRunner
from test_engine_plugin import COMMIT, _drift, context, inspect_report, lease

JOB = 'job-3d'


@pytest.fixture
def runner():
    fake = FakeRunner(inspect_report())
    yield fake
    fake.close()


@pytest.fixture
def plugin(runner):
    return AISmithEnginePlugin(MESH3D, engine_versions(MESH3D, inspect_report()), engine_url=runner.url)


def job_calls(runner):
    return runner.methods(JOB)


async def test_a_mesh_job_runs_through_the_runner(plugin, runner, tmp_path):
    ctx = context(tmp_path)
    output = await plugin.execute(ctx, plugin.validate(lease()))
    assert output.path.read_bytes() == GLB and output.path.is_relative_to(ctx.workspace / 'engine')
    assert (output.model_version, output.seed) == (COMMIT, 7)
    assert output.diagnostics['effective_inputs'] == {'resolution': 1024, 'seed': 7}
    assert output.diagnostics['runtime_versions'] == plugin.versions
    assert runner.methods()[0] == 'GET /inspect?full=0'
    assert job_calls(runner) == [f'PUT /jobs/{JOB}/inputs/front', f'POST /jobs/{JOB}/run',
                                 f'GET /jobs/{JOB}/output', f'DELETE /jobs/{JOB}']
    (spec,) = runner.runs
    # The profile's 1,800 s mesh deadline, less the time already spent.
    assert spec.pop('deadline_seconds') in (1799, 1800)
    assert spec == {'schema_version': 1, 'stage': 'mesh', 'inputs': ['front'],
                    'settings': {'resolution': 1024, 'seed': 7}}
    assert runner.inputs == {}  # the runner dropped them with the job


async def test_inputs_reach_the_runner_byte_for_byte(plugin, runner, tmp_path):
    ctx = context(tmp_path)
    seen = {}
    original = runner.server.RequestHandlerClass.do_PUT

    def spy(handler):
        original(handler)
        seen.update(runner.inputs.get(JOB, {}))

    runner.server.RequestHandlerClass.do_PUT = spy
    await plugin.execute(ctx, plugin.validate(lease()))
    assert seen == {'front': ctx.input_paths['inputs/pool/front.png'].read_bytes()}


@pytest.mark.parametrize('change', ['commit', 'unpinned', 'blender'])
async def test_drift_refuses_before_any_job_call(plugin, runner, tmp_path, change):
    runner.control['inspect'] = _drift(inspect_report(), change)
    with pytest.raises(EngineDrift):
        await plugin.execute(context(tmp_path), plugin.validate(lease()))
    assert job_calls(runner) == []


async def test_an_unreachable_runner_is_drift(tmp_path):
    plugin = AISmithEnginePlugin(MESH3D, engine_versions(MESH3D, inspect_report()),
                                 engine_url='http://127.0.0.1:9')
    with pytest.raises(EngineDrift, match='unreachable'):
        await plugin.execute(context(tmp_path), plugin.validate(lease()))


async def test_a_rejected_result_is_terminal(plugin, runner, tmp_path):
    runner.control['run'] = {'status': 'rejected', 'error': "input 'front' is not an image"}
    with pytest.raises(PluginRequestRejected, match='not an image'):
        await plugin.execute(context(tmp_path), plugin.validate(lease()))
    assert job_calls(runner)[-1] == f'DELETE /jobs/{JOB}'


async def test_a_failed_result_carries_the_engine_error(plugin, runner, tmp_path):
    runner.control['run'] = {'status': 'failed', 'error': 'deadline exceeded'}
    with pytest.raises(EngineFailed, match='deadline exceeded'):
        await plugin.execute(context(tmp_path), plugin.validate(lease()))
    assert job_calls(runner) == [f'PUT /jobs/{JOB}/inputs/front', f'POST /jobs/{JOB}/run', f'DELETE /jobs/{JOB}']


@pytest.mark.parametrize('run, error', [
    ({'output': {'sha256': '0' * 64}, 'x_sha256': '0' * 64}, 'size and sha256'),
    ({'x_sha256': '1' * 64}, 'X-Sha256 does not match'),
    ({'body': b'glTF' + b'\x01' * 124}, 'size and sha256'),
    ({'body': GLB + b'extra'}, 'Content-Length does not match'),
    ({'output': {'bytes': 64}}, 'Content-Length does not match'),
    ({'body': b'PK\x03\x04' + bytes(124), 'output': {'sha256': hashlib.sha256(b'PK\x03\x04' + bytes(124)).hexdigest()},
      'x_sha256': hashlib.sha256(b'PK\x03\x04' + bytes(124)).hexdigest()}, 'not a binary glTF'),
    ({'output': {'url': '/jobs/other/output'}}, 'output URL'),
    ({'output': {'content_type': 'model/obj'}}, 'not a GLB'),
    ({'status': 'done'}, "status 'done'"),
    ({'result': {'status': None}}, 'status None'),
    ({'result': {'stage': 'clean'}}, 'another stage'),
    ({'http_status': 409}, 'answered 409'),
])
async def test_a_bad_result_or_output_is_refused(plugin, runner, tmp_path, run, error):
    runner.control['run'] = run
    with pytest.raises(EngineFailed, match=error):
        await plugin.execute(context(tmp_path), plugin.validate(lease()))
    assert job_calls(runner)[-1] == f'DELETE /jobs/{JOB}'


async def test_a_result_from_another_engine_fails_closed(plugin, runner, tmp_path):
    runner.control['run'] = {'result': {'engine': {**inspect_report()['engine'], 'commit': 'e' * 40}}}
    with pytest.raises(EngineDrift, match='not the registered one'):
        await plugin.execute(context(tmp_path), plugin.validate(lease()))
    assert job_calls(runner)[-1] == f'DELETE /jobs/{JOB}'


async def test_the_deadline_cancels_the_runner_job(plugin, runner, tmp_path):
    runner.control['run'] = {'sleep': 300}
    started = time.monotonic()
    with pytest.raises(EngineTimedOut, match='deadline'):
        await plugin.execute(context(tmp_path, deadline_seconds=3), plugin.validate(lease()))
    assert time.monotonic() - started < 20
    assert runner.runs[0]['deadline_seconds'] <= 3
    assert job_calls(runner)[-2:] == [f'POST /jobs/{JOB}/cancel', f'DELETE /jobs/{JOB}']


async def test_lease_cancellation_cancels_the_runner_job(plugin, runner, tmp_path):
    runner.control['run'] = {'sleep': 300}
    cancel = asyncio.Event()
    task = asyncio.create_task(plugin.execute(context(tmp_path, cancel=cancel), plugin.validate(lease())))
    while f'POST /jobs/{JOB}/run' not in job_calls(runner):
        await asyncio.sleep(0.02)
    cancel.set()
    with pytest.raises(EngineCancelled):
        await task
    assert job_calls(runner)[-2:] == [f'POST /jobs/{JOB}/cancel', f'DELETE /jobs/{JOB}']


async def test_plugin_cancel_cancels_the_runner_job(plugin, runner, tmp_path):
    runner.control['run'] = {'sleep': 300}
    task = asyncio.create_task(plugin.execute(context(tmp_path), plugin.validate(lease())))
    while f'POST /jobs/{JOB}/run' not in job_calls(runner):
        await asyncio.sleep(0.02)
    assert await plugin.cancel(JOB) is True
    with pytest.raises(EngineCancelled):
        await task
    assert job_calls(runner)[-2:] == [f'POST /jobs/{JOB}/cancel', f'DELETE /jobs/{JOB}']
    assert await plugin.cancel(JOB) is False


async def test_our_own_cancellation_cancels_and_deletes(plugin, runner, tmp_path):
    runner.control['run'] = {'sleep': 300}
    task = asyncio.create_task(plugin.execute(context(tmp_path), plugin.validate(lease())))
    while f'POST /jobs/{JOB}/run' not in job_calls(runner):
        await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert f'POST /jobs/{JOB}/cancel' in job_calls(runner)
    assert job_calls(runner)[-1] == f'DELETE /jobs/{JOB}'


async def test_an_expired_context_calls_nothing(plugin, runner, tmp_path):
    with pytest.raises(EngineTimedOut):
        await plugin.execute(context(tmp_path, deadline_seconds=-1), plugin.validate(lease()))
    assert runner.calls == []


async def test_the_inspection_is_bounded_by_the_deadline(plugin, runner, tmp_path):
    runner.control['inspect_sleep'] = 30
    started = time.monotonic()
    # The inspect's own bound and the job's deadline race; either way it is the deadline.
    with pytest.raises(EngineTimedOut, match='while the engine was inspected|during the engine job'):
        await plugin.execute(context(tmp_path, deadline_seconds=2), plugin.validate(lease()))
    assert time.monotonic() - started < 15
    runner.control['inspect_sleep'] = 0


async def test_enrolment_inspects_fully_over_http(runner):
    settings = await prepare_engine_environment(MESH3D, {'OGWP_ENGINE_URL': runner.url + '/'})
    assert runner.methods() == ['GET /inspect?full=1']
    assert settings['OGWP_ENGINE_URL'] == runner.url
    assert 'MVS_ENGINE_COMMAND' not in settings


async def test_health_uses_the_runner(plugin, runner):
    assert (await plugin.health()).healthy
    runner.control['inspect'] = _drift(inspect_report(), 'commit')
    assert not (await plugin.health()).healthy


@pytest.mark.parametrize('url', ['http://8.8.8.8:8190', 'http://example.com:8190', 'http://1.1.1.1:8190',
                                 'ftp://127.0.0.1:8190', 'http://[2001:4860::8888]:8190', '127.0.0.1:8190'])
def test_a_non_private_runner_url_is_refused(url):
    with pytest.raises(ValueError, match='OGWP_ENGINE_URL must'):
        runner_url({'OGWP_ENGINE_URL': url})
    with pytest.raises(ValueError, match='OGWP_ENGINE_URL must'):
        AISmithEnginePlugin(MESH3D, engine_versions(MESH3D, inspect_report()), engine_url=url)


@pytest.mark.parametrize('url', ['http://127.0.0.1:8190', 'http://localhost:8190', 'http://10.0.0.5:8190',
                                 'http://192.168.1.20:8190/'])
def test_loopback_and_private_runner_urls_are_accepted(url):
    assert runner_url({'OGWP_ENGINE_URL': url}) == url.rstrip('/')
    assert runner_url({}) is None


async def test_a_runner_made_failure_carries_its_error(plugin, runner, tmp_path):
    """Deadline, cancel and crash failures arrive as result JSON with diagnostics.runner."""
    runner.control['run'] = {'status': 'failed', 'error': 'deadline exceeded',
                             'result': {'diagnostics': {'runner': {'returncode': -9, 'log_tail': 'killed'}}}}
    with pytest.raises(EngineFailed, match='deadline exceeded'):
        await plugin.execute(context(tmp_path), plugin.validate(lease()))
    assert job_calls(runner)[-1] == f'DELETE /jobs/{JOB}'


async def test_the_runner_deadline_stays_in_its_range(plugin, runner, tmp_path, monkeypatch):
    from outbound_gpu_worker_pool import engine

    stage = plugin.recipe.stage_for('mesh.trellis2.image_to_mesh.v1')
    monkeypatch.setattr(engine, 'MAX_RUNNER_DEADLINE_SECONDS', 2)
    await plugin.execute(context(tmp_path), plugin.validate(lease()))
    assert stage.deadline_seconds > 2 and runner.runs[0]['deadline_seconds'] == 2



# Transfers: one absolute deadline, and cancellation observed throughout.

async def _until(condition, within: float = 30.0) -> None:
    """Poll to a deadline, so a loaded machine slows the test rather than failing it."""
    deadline = time.monotonic() + within
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError('condition not reached')
        await asyncio.sleep(0.02)


async def test_a_trickling_output_cannot_outlast_the_deadline(plugin, runner, tmp_path):
    # 128 bytes, 16 at a time, 2 s apart: about 16 s, every chunk well inside any
    # per-operation timeout.
    runner.control['run'] = {'trickle': 2}
    started = time.monotonic()
    with pytest.raises(EngineTimedOut, match='deadline'):
        await plugin.execute(context(tmp_path, deadline_seconds=4), plugin.validate(lease()))
    assert time.monotonic() - started < 12
    assert f'GET /jobs/{JOB}/output' in job_calls(runner)
    assert job_calls(runner)[-1] == f'DELETE /jobs/{JOB}'


@pytest.mark.parametrize('how', ['plugin', 'lease'])
async def test_cancelling_during_upload_stops_before_run(plugin, runner, tmp_path, how):
    runner.control['run'] = {'put_sleep': 20}
    cancel = asyncio.Event()
    task = asyncio.create_task(plugin.execute(context(tmp_path, cancel=cancel), plugin.validate(lease())))
    await _until(lambda: f'PUT /jobs/{JOB}/inputs/front' in job_calls(runner))
    started = time.monotonic()
    if how == 'plugin':
        assert await plugin.cancel(JOB) is True
    else:
        cancel.set()
    with pytest.raises(EngineCancelled):
        await task
    assert time.monotonic() - started < 10
    await _until(lambda: f'DELETE /jobs/{JOB}' in job_calls(runner))
    assert f'POST /jobs/{JOB}/run' not in job_calls(runner)
    assert job_calls(runner)[-1] == f'DELETE /jobs/{JOB}'


@pytest.mark.parametrize('how', ['plugin', 'lease'])
async def test_cancelling_during_download_returns_no_output(plugin, runner, tmp_path, how):
    runner.control['run'] = {'trickle': 1}
    cancel = asyncio.Event()
    task = asyncio.create_task(plugin.execute(context(tmp_path, cancel=cancel), plugin.validate(lease())))
    await _until(lambda: f'GET /jobs/{JOB}/output' in job_calls(runner))
    started = time.monotonic()
    if how == 'plugin':
        assert await plugin.cancel(JOB) is True
    else:
        cancel.set()
    with pytest.raises(EngineCancelled):
        await task
    assert time.monotonic() - started < 6
    assert job_calls(runner)[-1] == f'DELETE /jobs/{JOB}'


async def test_a_job_cancelled_before_it_starts_calls_no_stage(plugin, runner, tmp_path):
    cancel = asyncio.Event()
    cancel.set()
    with pytest.raises(EngineCancelled):
        await plugin.execute(context(tmp_path, cancel=cancel), plugin.validate(lease()))
    assert f'POST /jobs/{JOB}/run' not in job_calls(runner)
