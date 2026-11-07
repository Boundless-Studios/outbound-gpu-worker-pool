"""The standalone engine worker: `agent_main` with an engine profile named alone."""
import json
import shlex
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import pytest

from outbound_gpu_worker_pool import (
    JobRequirements, JobStatus, JobSubmission, MemoryAssetStore, MemoryAuditLog, MemoryJobStore,
    MemoryWorkerAuthenticator, MemoryWorkerRegistry, WorkerEnrollment, WorkerIdentity, agent_main,
)
from outbound_gpu_worker_pool.agent import AgentOutcome
from outbound_gpu_worker_pool.coordinator import create_coordinator_app
from outbound_gpu_worker_pool.engine import (
    ENGINE_COMMAND_VARIABLE, ENGINE_URL_VARIABLE, AISmithEnginePlugin, EngineDrift, engine_versions,
)
from outbound_gpu_worker_pool.engine_recipes import MESH3D, MESH3D_CLEAN_CAPABILITY, MESH3D_MESH_CAPABILITY
from outbound_gpu_worker_pool.machine_profiles import CONFIG_KEY, RECIPE_KEY, STANDARD
from outbound_gpu_worker_pool.memory import MemoryAssetTransfer
from outbound_gpu_worker_pool.plugins import capability_schemas_from_plugins
from outbound_gpu_worker_pool.service import WorkerPoolService
from fake_engine_runner import FakeRunner
from test_agent_main import STATIC_ENVIRONMENT
from test_engine_plugin import FakeEngine, _drift, inspect_report

ROOTS = {'TRELLIS_REFINER_DIR': '/models/trellis', 'BLENDER_EXE': '/opt/blender/blender',
         'U2NET_HOME': '/models/u2net', 'UNIRIG_DIR': '/models/unirig', 'AISMITH_DATA_DIR': '/data/aismith'}


@pytest.fixture
def engine(tmp_path):
    directory = tmp_path / 'engine'
    directory.mkdir()
    return FakeEngine(directory)


@pytest.fixture
def runner():
    fake = FakeRunner(inspect_report())
    yield fake
    fake.close()


def engine_environment(engine, tmp_path, **extra):
    return STATIC_ENVIRONMENT | {'MVS_WORKER_PROFILE': MESH3D,
                                 ENGINE_COMMAND_VARIABLE: shlex.join(engine.command),
                                 'OGWP_WORKER_WORKSPACE': str(tmp_path / 'workspace'), **extra}


class _NoGpuSampler:
    def sample(self):
        return ()


@asynccontextmanager
async def coordinator(plugin):
    """The real coordinator app over in-memory stores, as the agent tests use."""
    jobs, assets = MemoryJobStore(), MemoryAssetStore()
    service = WorkerPoolService(
        jobs, assets, MemoryWorkerRegistry(), MemoryAuditLog(),
        MemoryWorkerAuthenticator({'token-a': WorkerIdentity('worker-a', 'static:worker-a', 'static')}),
        capability_schemas_from_plugins((plugin,)),
        enrollments={'worker-a': WorkerEnrollment('static:worker-a', None)},
        pre_auth_limit_per_minute=100_000, per_source_limit_per_minute=100_000,
        per_worker_limit_per_minute=100_000, global_limit_per_minute=100_000)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_coordinator_app(service)),
                                 base_url='http://coordinator') as http:
        yield service, jobs, assets, http


async def run_one_mesh_job(prepared, plugin):
    """Build the exclusive agent, lease one pinned mesh job, and return the settled record."""
    agent = agent_main.build_agent_from_env(prepared, profile_plugin=plugin, exclusive=True)
    versions = json.loads(prepared['OGWP_WORKER_RUNTIME_VERSIONS'])
    async with coordinator(plugin) as (service, jobs, assets, http):
        agent._http, agent._transfer, agent._gpu_sampler = http, MemoryAssetTransfer(assets), _NoGpuSampler()
        await assets.write_once('inputs/pool/front.png', b'\x89PNG front', 'image/png')
        submission = JobSubmission(
            job_id=str(uuid4()), idempotency_key='pool:mesh3d', capability_id=MESH3D_MESH_CAPABILITY,
            input_keys=('inputs/pool/front.png',), output_key='outputs/pool/mesh.glb',
            payload={'inputs': {'front': 'inputs/pool/front.png'}, 'resolution': 1024, 'seed': 7},
            requirements=JobRequirements(runtime_versions={RECIPE_KEY: versions[RECIPE_KEY],
                                                           CONFIG_KEY: versions[CONFIG_KEY]}),
            execution_deadline_seconds=1800)
        await service.submit(submission)
        assert await agent.run_once() is AgentOutcome.COMPLETED
        record = await jobs.get(submission.job_id)
        assert (await assets.read(record.output_key)).startswith(b'glTF')
        return record


# Startup.

async def test_startup_inspects_fully_and_registers_only_the_engine(engine, tmp_path):
    prepared, plugin = await agent_main.prepare_engine_profile(engine_environment(engine, tmp_path))
    assert engine.calls() == [['inspect', '--full']]
    assert isinstance(plugin, AISmithEnginePlugin) and plugin.transport == 'local'
    agent = agent_main.build_agent_from_env(prepared, profile_plugin=plugin, exclusive=True)
    assert agent.capability_ids == (MESH3D_MESH_CAPABILITY, MESH3D_CLEAN_CAPABILITY)
    assert set(agent._plugins.values()) == {plugin}
    assert agent._runtime_versions == engine_versions(MESH3D, inspect_report()) == plugin.versions
    assert prepared['OGWP_WORKER_CONCURRENCY'] == '1'
    assert (prepared['OGWP_WORKER_VRAM_MB'], prepared['OGWP_WORKER_GPU_MODEL']) == ('24564', 'NVIDIA GeForce RTX 4090')


async def test_drift_at_startup_starts_no_agent(engine, tmp_path, monkeypatch):
    engine.control['inspect'] = _drift(inspect_report(), 'commit')
    engine.save()
    monkeypatch.setattr(agent_main, 'build_agent_from_env', lambda *a, **k: pytest.fail('an agent was built'))
    with pytest.raises(EngineDrift, match='engine commit'):
        await agent_main.main(engine_environment(engine, tmp_path))


@pytest.mark.parametrize('profiles', [f'{MESH3D},{STANDARD}', f'{STANDARD},{MESH3D}'])
async def test_the_engine_profile_beside_another_is_refused(engine, tmp_path, profiles):
    with pytest.raises(ValueError, match=f'{MESH3D} is an engine profile and runs alone'):
        await agent_main.prepare_engine_profile(engine_environment(engine, tmp_path, MVS_WORKER_PROFILE=profiles))
    with pytest.raises(ValueError, match='runs alone'):
        await agent_main.main(engine_environment(engine, tmp_path, MVS_WORKER_PROFILE=profiles))
    assert engine.calls() == []


@pytest.mark.parametrize('plugins', ['comfy-workflow', 'aismith-engine,comfy-workflow',
                                     'deterministic-echo', 'aismith-engine,deterministic-echo'])
async def test_other_plugins_are_refused(engine, tmp_path, plugins):
    with pytest.raises(ValueError, match='runs only the aismith-engine plugin'):
        await agent_main.prepare_engine_profile(engine_environment(engine, tmp_path, OGWP_WORKER_PLUGINS=plugins))
    assert engine.calls() == []


async def test_only_the_engine_plugin_name_is_allowed(engine, tmp_path):
    prepared, _ = await agent_main.prepare_engine_profile(
        engine_environment(engine, tmp_path, OGWP_WORKER_PLUGINS='aismith-engine'))
    assert prepared['OGWP_WORKER_CONCURRENCY'] == '1'


async def test_concurrency_other_than_one_is_refused(engine, tmp_path):
    with pytest.raises(ValueError, match='OGWP_WORKER_CONCURRENCY must be 1'):
        await agent_main.prepare_engine_profile(engine_environment(engine, tmp_path, OGWP_WORKER_CONCURRENCY='2'))


async def test_comfy_profiles_take_no_engine_path():
    assert await agent_main.prepare_engine_profile(STATIC_ENVIRONMENT | {'MVS_WORKER_PROFILE': STANDARD}) is None
    assert await agent_main.prepare_engine_profile(STATIC_ENVIRONMENT) is None


async def test_main_runs_the_exclusive_engine_agent(engine, tmp_path, monkeypatch):
    built = []

    class _Agent:
        async def run_forever(self, stop):
            return None

    def build(environment, profile_plugin=None, *, exclusive=False):
        built.append((profile_plugin, exclusive))
        return _Agent()

    monkeypatch.setattr(agent_main, 'build_agent_from_env', build)
    assert await agent_main.main(engine_environment(engine, tmp_path)) == 0
    (plugin, exclusive), = built
    assert isinstance(plugin, AISmithEnginePlugin) and exclusive


def test_an_exclusive_agent_needs_its_plugin():
    with pytest.raises(ValueError, match='needs its profile plugin'):
        agent_main.build_agent_from_env(STATIC_ENVIRONMENT, exclusive=True)


async def test_the_engine_always_runs_offline_with_the_runtime_roots(engine, tmp_path, monkeypatch):
    for key, value in {**ROOTS, 'AISMITH_OFFLINE': '0'}.items():
        monkeypatch.setenv(key, value)
    await agent_main.prepare_engine_profile(engine_environment(engine, tmp_path))
    seen = json.loads((engine.directory / 'env.json').read_text())
    assert seen == {**ROOTS, 'AISMITH_OFFLINE': '1'}


async def test_timeouts_come_from_the_environment(engine, tmp_path):
    prepared, plugin = await agent_main.prepare_engine_profile(engine_environment(
        engine, tmp_path, MVS_ENGINE_INSPECT_TIMEOUT_SECONDS='90', MVS_ENGINE_KILL_GRACE_SECONDS='3'))
    assert (plugin.inspect_timeout_seconds, plugin.kill_grace_seconds) == (90.0, 3.0)
    with pytest.raises(ValueError, match='MVS_ENGINE_FULL_INSPECT_TIMEOUT_SECONDS must be a positive'):
        await agent_main.prepare_engine_profile(engine_environment(
            engine, tmp_path, MVS_ENGINE_FULL_INSPECT_TIMEOUT_SECONDS='0'))


# The whole loop: lease, run the engine, upload, complete with attestation.

async def test_the_local_engine_worker_completes_a_mesh_job(engine, tmp_path):
    prepared, plugin = await agent_main.prepare_engine_profile(engine_environment(engine, tmp_path))
    record = await run_one_mesh_job(prepared, plugin)
    assert record.status is JobStatus.COMPLETED
    diagnostics = record.execution_metadata['diagnostics']
    assert diagnostics['effective_inputs'] == {'resolution': 1024, 'seed': 7}
    assert diagnostics['runtime_versions'] == plugin.versions
    assert [call[:2] for call in engine.calls()] == [['inspect', '--full'], ['inspect'], ['run', 'mesh']]


async def test_the_sidecar_engine_worker_completes_a_mesh_job(runner, tmp_path):
    environment = STATIC_ENVIRONMENT | {'MVS_WORKER_PROFILE': MESH3D, ENGINE_URL_VARIABLE: runner.url,
                                        'OGWP_WORKER_WORKSPACE': str(tmp_path / 'workspace')}
    prepared, plugin = await agent_main.prepare_engine_profile(environment)
    assert plugin.transport == 'http' and prepared[ENGINE_URL_VARIABLE] == runner.url
    assert ENGINE_COMMAND_VARIABLE not in prepared
    record = await run_one_mesh_job(prepared, plugin)
    assert record.status is JobStatus.COMPLETED
    assert record.execution_metadata['diagnostics']['effective_inputs'] == {'resolution': 1024, 'seed': 7}
    job = record.job_id
    assert runner.methods()[0] == 'GET /inspect?full=1'
    assert runner.methods(job) == [f'PUT /jobs/{job}/inputs/front', f'POST /jobs/{job}/run',
                                   f'GET /jobs/{job}/output', f'DELETE /jobs/{job}']
    assert runner.runs[0]['deadline_seconds'] <= 1800
    assert not any(path.startswith('/jobs') and method == 'POST' and path.endswith('/cancel')
                   for method, path in runner.calls)
