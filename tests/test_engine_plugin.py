"""The `aismith-engine` plugin against a fake engine executable: no real engine, no GPU."""
import asyncio
import json
import os
import shlex
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from outbound_gpu_worker_pool import (
    AssetGrant, JobRequirements, JobStatus, LeaseGrant, WorkerRegistration,
)
from outbound_gpu_worker_pool.engine import (
    ENGINE_COMMAND_VARIABLE, AISmithEnginePlugin, EngineCancelled, EngineDrift, EngineFailed,
    EngineTimedOut, engine_versions,
)
from outbound_gpu_worker_pool.engine_recipes import (
    MESH3D, MESH3D_CLEAN_CAPABILITY, MESH3D_MESH_CAPABILITY,
)
from outbound_gpu_worker_pool.machine_profiles import (
    CONFIG_KEY, PROFILES, RECIPE_KEY, describe_profile, prepare_environment, recipe_key,
)
from outbound_gpu_worker_pool.plugins import ExecutionContext, PluginRequestRejected
from test_engine_profile import qualified_profile
from test_service import WORKER_A, _harness, _manifest, _submission

FAKE_ENGINE = r'''
import hashlib, json, subprocess, sys, time
from pathlib import Path

here = Path(__file__).parent
control = json.loads((here / 'control.json').read_text())
args = sys.argv[1:]
with (here / 'calls.jsonl').open('a') as log:
    log.write(json.dumps(args) + '\n')
if args[0] == 'inspect':
    print(json.dumps(control['inspect']))
    sys.exit(control.get('inspect_exit', 0))
stage = args[1]
spec = Path(args[args.index('--spec') + 1])
workdir = Path(args[args.index('--workdir') + 1])
run = control.get('run', {})
(here / 'last-spec.json').write_text(spec.read_text())
if run.get('child'):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
    (here / 'child.pid').write_text(str(child.pid))
if run.get('sleep'):
    time.sleep(run['sleep'])
if run.get('crash') is not None:
    sys.exit(run['crash'])
data = json.loads(spec.read_text())
status = run.get('status', 'ok')
result = {'stage': stage, 'status': status, 'schema_version': 1, 'output': None, 'gate': None,
          'effective_inputs': {}, 'measured': {}, 'engine': control['inspect']['engine'],
          'weights': {'ok': True, 'pinned': True, 'files': []}, 'diagnostics': {},
          'timings_s': {'total': 1.5}, 'error': run.get('error')}
if status == 'ok':
    output = workdir / (stage + '.glb')
    body = b'glTF' + bytes(124)
    output.write_bytes(body)
    result['output'] = {'path': str(output), 'content_type': 'model/gltf-binary',
                        'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body),
                        **run.get('output', {})}
    result['effective_inputs'] = run.get('effective_inputs',
                                         {**data['settings'], 'views': sorted(data['inputs'])})
    result['gate'] = {'name': 's2-canonicalize', 'passed': True} if stage == 'clean' else None
    result['measured'] = run.get('measured', {'triangles': 1234, 'watertight': True})
result.update(run.get('result', {}))
(workdir / 'result.json').write_text(json.dumps(result))
print(json.dumps(result))
sys.exit(run.get('exit', {'ok': 0, 'failed': 1, 'rejected': 2}[status]))
'''


def inspect_report() -> dict:
    return {
        'engine': {'name': 'local-3d-studio', 'version': '0.1.0', 'commit': 'a' * 40,
                   'sources_lock_sha256': 'b' * 64, 'weights_lock_sha256': 'c' * 64},
        'gpus': [{'name': 'NVIDIA GeForce RTX 4090', 'compute_capability': '8.9',
                  'memory_mb': 24564, 'driver': '581.57'}],
        'runtimes': {
            'trellis': {'python': '/opt/trellis/python', 'torch': '2.9.0+cu128', 'cuda': '12.8',
                        'native_contract': {'cuda_architecture': '8.9'}, 'comfyui_version': '0.3.60'},
            'unirig': {'torch': None},
            'blender': {'path': '/opt/blender/blender', 'version': 'Blender 4.2.23 LTS'},
        },
        'weights': {'ok': True, 'pinned': True, 'files': []},
        'offline': True,
    }


class FakeEngine:
    def __init__(self, directory: Path):
        self.directory = directory
        (directory / 'fake_engine.py').write_text(FAKE_ENGINE)
        self.command = (sys.executable, str(directory / 'fake_engine.py'))
        self.control = {'inspect': inspect_report(), 'run': {}}
        self.save()

    def save(self):
        (self.directory / 'control.json').write_text(json.dumps(self.control))

    def set(self, **run):
        self.control['run'] = run
        self.save()

    def calls(self) -> list[list[str]]:
        path = self.directory / 'calls.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def spec(self) -> dict:
        return json.loads((self.directory / 'last-spec.json').read_text())


@pytest.fixture
def qualified(monkeypatch):
    monkeypatch.setitem(PROFILES, MESH3D, qualified_profile())


@pytest.fixture
def engine(tmp_path, qualified):
    directory = tmp_path / 'engine'
    directory.mkdir()
    return FakeEngine(directory)


@pytest.fixture
def plugin(engine):
    return AISmithEnginePlugin(MESH3D, engine_versions(MESH3D, inspect_report()), command=engine.command,
                               kill_grace_seconds=1)


def lease(capability=MESH3D_MESH_CAPABILITY, payload=None, input_keys=('inputs/pool/front.png',)):
    expires = datetime.now(UTC) + timedelta(minutes=15)
    return LeaseGrant(
        job_id='job-3d', claim_token='claim-1', lease_until=expires, execution_deadline_seconds=7200,
        capability_id=capability, contract_version=1, request_digest='digest', idempotency_key='idem-1',
        input_keys=tuple(input_keys), output_key='outputs/pool/mesh.glb',
        payload=payload if payload is not None else {
            'inputs': {'front': 'inputs/pool/front.png'}, 'resolution': 1024, 'seed': 7},
        input_grants=(), output_grant=AssetGrant(key='outputs/pool/mesh.glb', url='memory://upload/x',
                                                 method='PUT', expires_at=expires))


def context(tmp_path, keys=('inputs/pool/front.png',), *, deadline_seconds=7200, cancel=None):
    workspace = tmp_path / 'work'
    (workspace / 'inputs').mkdir(parents=True, exist_ok=True)
    paths = {}
    for key in keys:
        paths[key] = workspace / 'inputs' / Path(key).name
        paths[key].write_bytes(b'input')
    progress = []

    async def report(percent):
        progress.append(percent)

    return ExecutionContext(job_id='job-3d', lease_id='claim-1',
                            deadline=datetime.now(UTC) + timedelta(seconds=deadline_seconds),
                            workspace=workspace, input_paths=paths, cancel=cancel or asyncio.Event(),
                            progress=report)


# Capabilities and validation.

def test_the_manifest_lists_both_capabilities(plugin):
    manifest = plugin.capabilities()
    assert [c.capability_id for c in manifest.capabilities] == [MESH3D_MESH_CAPABILITY, MESH3D_CLEAN_CAPABILITY]
    schemas = {s.capability_id: s.input_schema for s in manifest.schemas}
    assert schemas[MESH3D_MESH_CAPABILITY]['properties']['resolution'] == {'enum': [512, 1024]}
    assert schemas[MESH3D_CLEAN_CAPABILITY]['oneOf'] == [{'required': ['quality_tier']},
                                                         {'required': ['target_triangles']}]


def test_a_mesh_request_validates(plugin):
    request = plugin.validate(lease())
    assert request.inputs == {'resolution': 1024, 'seed': 7, 'inputs': {'front': 'inputs/pool/front.png'}}
    assert request.seed == 7


FRONT = {'front': 'inputs/pool/front.png'}


@pytest.mark.parametrize('payload, keys, error', [
    ({'inputs': FRONT, 'resolution': 768, 'seed': 7}, None, 'resolution must be one of'),
    ({'inputs': FRONT, 'resolution': 1024}, None, 'seed is required'),
    ({'inputs': FRONT, 'resolution': 1024, 'seed': True}, None, 'seed must be an integer'),
    ({'inputs': FRONT, 'resolution': 1024, 'seed': -1}, None, 'seed must be an integer'),
    ({'inputs': FRONT, 'resolution': 1024, 'seed': 7, 'steps': 8}, None, 'unsupported keys'),
    ({'inputs': {}, 'resolution': 1024, 'seed': 7}, None, 'input front is required'),
    ({'inputs': {**FRONT, 'top': 'inputs/pool/front.png'}, 'resolution': 1024, 'seed': 7}, None, 'does not take'),
    ({'inputs': {'front': 'inputs/pool/other.png'}, 'resolution': 1024, 'seed': 7}, None, 'granted input'),
    ({'inputs': FRONT, 'resolution': 1024, 'seed': 7}, ('inputs/pool/front.png', 'inputs/pool/back.png'),
     'every granted input'),
])
def test_bad_mesh_requests_are_rejected(plugin, payload, keys, error):
    with pytest.raises(PluginRequestRejected, match=error):
        plugin.validate(lease(payload=payload, input_keys=keys or ('inputs/pool/front.png',)))


MESH = {'mesh': 'inputs/pool/raw.glb'}


@pytest.mark.parametrize('payload, error', [
    ({'inputs': MESH, 'quality_tier': 'high'}, None),
    ({'inputs': MESH, 'target_triangles': 120000}, None),
    ({'inputs': MESH}, 'exactly one of'),
    ({'inputs': MESH, 'quality_tier': 'high', 'target_triangles': 120000}, 'exactly one of'),
    ({'inputs': MESH, 'quality_tier': 'extreme'}, 'quality_tier must be one of'),
    ({'inputs': MESH, 'target_triangles': 999}, 'target_triangles must be an integer'),
    ({'inputs': {'front': 'inputs/pool/raw.glb'}, 'quality_tier': 'high'}, 'input mesh is required'),
])
def test_clean_requests_need_exactly_one_density(plugin, payload, error):
    grant = lease(MESH3D_CLEAN_CAPABILITY, payload, ('inputs/pool/raw.glb',))
    if error is None:
        assert plugin.validate(grant).seed is None
    else:
        with pytest.raises(PluginRequestRejected, match=error):
            plugin.validate(grant)


def test_other_capabilities_and_contracts_are_rejected(plugin):
    with pytest.raises(PluginRequestRejected, match='unsupported capability'):
        plugin.validate(lease('video.minimax_h3.text_to_video.v1'))
    with pytest.raises(PluginRequestRejected, match='contract version'):
        plugin.validate(replace(lease(), contract_version=2))


def test_the_plugin_refuses_versions_of_another_recipe(engine):
    with pytest.raises(ValueError, match='not the mesh3d-24gb-standard recipe'):
        AISmithEnginePlugin(MESH3D, {**engine_versions(MESH3D, inspect_report()), RECIPE_KEY: 'f' * 64},
                            command=engine.command)


# Execution.

async def test_a_mesh_job_produces_one_attested_glb(plugin, engine, tmp_path):
    ctx = context(tmp_path)
    output = await plugin.execute(ctx, plugin.validate(lease()))
    assert output.path.read_bytes().startswith(b'glTF')
    assert output.path.is_relative_to(ctx.workspace / 'engine')
    assert output.content_type == 'model/gltf-binary'
    assert (output.model_id, output.model_version, output.seed) == ('aismith-engine/mesh', 'a' * 40, 7)
    diagnostics = output.diagnostics
    assert diagnostics['runtime_versions'] == plugin.versions
    assert diagnostics['effective_inputs'] == {'resolution': 1024, 'seed': 7}
    assert diagnostics['engine_effective_inputs']['views'] == ['front']
    assert diagnostics['measured'] == {'triangles': 1234, 'watertight': True}
    assert diagnostics['engine']['commit'] == 'a' * 40
    assert diagnostics['weights'] == {'ok': True, 'pinned': True}
    # The spec binds the downloaded file to its stage input name.
    assert engine.spec() == {'schema_version': 1, 'stage': 'mesh', 'settings': {'resolution': 1024, 'seed': 7},
                             'inputs': {'front': str(ctx.input_paths['inputs/pool/front.png'])}}
    # A cached-hash inspect ran before the stage, never --full per job.
    calls = engine.calls()
    assert calls[0] == ['inspect'] and calls[1][:2] == ['run', 'mesh']


async def test_a_clean_job_attests_only_what_it_set(plugin, engine, tmp_path):
    engine.set(effective_inputs={'method': 'voxel', 'quality_tier': 'high', 'target_triangles': 80000})
    payload = {'inputs': MESH, 'quality_tier': 'high'}
    output = await plugin.execute(context(tmp_path, ('inputs/pool/raw.glb',)),
                                  plugin.validate(lease(MESH3D_CLEAN_CAPABILITY, payload, ('inputs/pool/raw.glb',))))
    assert output.diagnostics['effective_inputs'] == {'quality_tier': 'high'}
    assert output.diagnostics['engine_effective_inputs']['target_triangles'] == 80000
    assert output.diagnostics['gate'] == {'name': 's2-canonicalize', 'passed': True}
    assert output.seed is None


async def test_a_further_profile_attests_under_namespaced_keys(engine, tmp_path):
    versions = engine_versions(MESH3D, inspect_report())
    plugin = AISmithEnginePlugin(MESH3D, versions, command=engine.command, namespaced=True)
    output = await plugin.execute(context(tmp_path), plugin.validate(lease()))
    reported = output.diagnostics['runtime_versions']
    assert RECIPE_KEY not in reported
    assert reported[recipe_key(MESH3D)] == versions[RECIPE_KEY]
    assert reported[recipe_key(MESH3D, CONFIG_KEY)] == versions[CONFIG_KEY]
    assert reported['blender'] == 'Blender 4.2.23 LTS'


async def test_a_rejected_spec_is_terminal(plugin, engine, tmp_path):
    engine.set(status='rejected', error="input 'front' is not an image")
    with pytest.raises(PluginRequestRejected, match='not an image'):
        await plugin.execute(context(tmp_path), plugin.validate(lease()))


async def test_a_failed_stage_carries_the_engine_error(plugin, engine, tmp_path):
    engine.set(status='failed', error='weights: trellis2-4b-fp8 missing')
    with pytest.raises(EngineFailed, match='trellis2-4b-fp8 missing'):
        await plugin.execute(context(tmp_path), plugin.validate(lease()))


@pytest.mark.parametrize('run, error', [
    ({'crash': 139}, 'exited 139 without a readable result.json'),
    ({'exit': 0, 'status': 'failed'}, "exited 0 with status 'failed'"),
    ({'result': {'stage': 'clean'}}, 'another stage'),
    ({'output': {'sha256': '0' * 64}}, 'size and sha256'),
    ({'output': {'path': '/etc/hosts'}}, 'inside its workdir'),
    ({'output': {'content_type': 'model/obj'}}, 'not a GLB'),
])
async def test_an_unusable_result_fails_the_job(plugin, engine, tmp_path, run, error):
    engine.set(**run)
    with pytest.raises(EngineFailed, match=error):
        await plugin.execute(context(tmp_path), plugin.validate(lease()))


async def test_a_result_from_another_engine_fails_closed(plugin, engine, tmp_path):
    engine.set(result={'engine': {**inspect_report()['engine'], 'commit': 'e' * 40}})
    with pytest.raises(EngineDrift, match='not the registered one'):
        await plugin.execute(context(tmp_path), plugin.validate(lease()))


def _drift(report, change):
    if change == 'commit':
        report['engine']['commit'] = 'e' * 40
    elif change == 'weights-lock':
        report['engine']['weights_lock_sha256'] = 'e' * 64
    elif change == 'unpinned':
        report['weights']['pinned'] = False
    elif change == 'gpu':
        report['gpus'][0]['compute_capability'] = '8.6'
    elif change == 'vram':
        report['gpus'][0]['memory_mb'] = 16376
    elif change == 'blender':
        report['runtimes']['blender']['version'] = 'Blender 4.3.2'
    elif change == 'native':
        report['runtimes']['trellis']['native_contract'] = None
    elif change == 'online':
        report['offline'] = False
    elif change == 'torch':
        report['runtimes']['trellis']['torch'] = '2.9.1+cu128'
    return report


@pytest.mark.parametrize('change', ['commit', 'weights-lock', 'unpinned', 'gpu', 'vram', 'blender',
                                    'native', 'online', 'torch'])
async def test_drift_since_registration_fails_before_the_stage_runs(plugin, engine, tmp_path, change):
    engine.control['inspect'] = _drift(inspect_report(), change)
    engine.save()
    with pytest.raises(EngineDrift):
        await plugin.execute(context(tmp_path), plugin.validate(lease()))
    assert [call[0] for call in engine.calls()] == ['inspect']


async def test_a_failing_inspect_fails_closed(plugin, engine, tmp_path):
    engine.control['inspect_exit'] = 1
    engine.save()
    with pytest.raises(EngineDrift, match='inspect exited 1'):
        await plugin.execute(context(tmp_path), plugin.validate(lease()))


def _gone(pid: int, within: float = 10.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        # A killed child is a zombie until its new parent reaps it.
        try:
            if os.waitpid(pid, os.WNOHANG) != (0, 0):
                return True
        except ChildProcessError:
            pass
        time.sleep(0.05)
    return False


async def _wait_for(path: Path):
    for _ in range(200):
        if path.exists() and path.read_text():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f'{path} never appeared')


async def test_the_deadline_kills_the_whole_process_group(plugin, engine, tmp_path):
    engine.set(child=True, sleep=300)
    started = time.monotonic()
    with pytest.raises(EngineTimedOut, match='deadline'):
        await plugin.execute(context(tmp_path, deadline_seconds=2), plugin.validate(lease()))
    assert time.monotonic() - started < 30
    assert _gone(int((engine.directory / 'child.pid').read_text()))


async def test_cancel_kills_the_whole_process_group(plugin, engine, tmp_path):
    engine.set(child=True, sleep=300)
    cancel = asyncio.Event()
    task = asyncio.create_task(plugin.execute(context(tmp_path, cancel=cancel), plugin.validate(lease())))
    await _wait_for(engine.directory / 'child.pid')
    cancel.set()
    with pytest.raises(EngineCancelled):
        await task
    assert _gone(int((engine.directory / 'child.pid').read_text()))


async def test_plugin_cancel_kills_a_running_job(plugin, engine, tmp_path):
    engine.set(child=True, sleep=300)
    task = asyncio.create_task(plugin.execute(context(tmp_path), plugin.validate(lease())))
    await _wait_for(engine.directory / 'child.pid')
    assert await plugin.cancel('job-3d') is True
    with pytest.raises(EngineFailed):
        await task
    assert _gone(int((engine.directory / 'child.pid').read_text()))
    assert await plugin.cancel('job-3d') is False


async def test_health_reports_drift(plugin, engine):
    assert (await plugin.health()).healthy
    engine.control['inspect'] = _drift(inspect_report(), 'commit')
    engine.save()
    assert not (await plugin.health()).healthy


# Enrolment and the coordinator, end to end.

async def test_enrolment_registers_what_each_job_rechecks(engine):
    settings = await prepare_environment({'MVS_WORKER_PROFILE': MESH3D,
                                          ENGINE_COMMAND_VARIABLE: shlex.join(engine.command)})
    versions = json.loads(settings['OGWP_WORKER_RUNTIME_VERSIONS'])
    assert versions == engine_versions(MESH3D, inspect_report())
    assert versions[RECIPE_KEY] == describe_profile(MESH3D)['recipe_version']
    assert settings['OGWP_WORKER_VRAM_MB'] == '24564'
    assert settings['OGWP_WORKER_GPU_MODEL'] == 'NVIDIA GeForce RTX 4090'
    assert settings['OGWP_WORKER_CONCURRENCY'] == '1'
    assert engine.calls() == [['inspect', '--full']]


async def test_enrolment_refuses_a_runtime_off_the_recipe(engine):
    engine.control['inspect'] = _drift(inspect_report(), 'unpinned')
    engine.save()
    with pytest.raises(EngineDrift, match='pinned'):
        await prepare_environment({'MVS_WORKER_PROFILE': MESH3D,
                                   ENGINE_COMMAND_VARIABLE: shlex.join(engine.command)})


@pytest.mark.parametrize('tamper', [None, 'seed', 'config'])
async def test_the_coordinator_accepts_a_real_engine_completion(plugin, engine, tmp_path, tamper):
    from outbound_gpu_worker_pool.service import CompletionRejected

    harness = _harness()
    pinned = {RECIPE_KEY: plugin.versions[RECIPE_KEY], CONFIG_KEY: plugin.versions[CONFIG_KEY]}
    submission = _submission('mesh3d', capability_id=MESH3D_MESH_CAPABILITY,
                             input_keys=('inputs/pool/front.png',), output_key='outputs/pool/mesh.glb',
                             payload={'inputs': FRONT, 'resolution': 1024, 'seed': 7},
                             requirements=JobRequirements(runtime_versions=pinned))
    await harness.assets.write_once('inputs/pool/front.png', b'png', 'image/png')
    await harness.service.submit(submission)
    await harness.service.register_heartbeat(WORKER_A, WorkerRegistration(
        worker_id=WORKER_A.worker_id, runtime_versions=plugin.versions,
        capabilities=plugin.capabilities().capabilities))
    grant = await harness.service.lease(WORKER_A, (MESH3D_MESH_CAPABILITY,))
    if tamper == 'seed':
        engine.set(effective_inputs={'resolution': 1024, 'seed': 8})
    output = await plugin.execute(context(tmp_path), plugin.validate(grant))
    diagnostics = output.diagnostics
    if tamper == 'config':
        diagnostics = {**diagnostics, 'runtime_versions': {**diagnostics['runtime_versions'], CONFIG_KEY: 'd' * 64}}
    body = output.path.read_bytes()
    await harness.assets.write_once(grant.output_key, body, output.content_type)
    complete = harness.service.complete(WORKER_A, grant.job_id, grant.claim_token, _manifest(
        grant, body, content_type=output.content_type, plugin_id='aismith-engine',
        model_id=output.model_id, model_version=output.model_version, diagnostics=diagnostics))
    if tamper is None:
        await complete
        record = harness.jobs.records[grant.job_id]
        assert record.status is JobStatus.COMPLETED
    else:
        with pytest.raises(CompletionRejected):
            await complete
