"""The dialogue template: voice references bind through audio slots.

A dialogue take anchors each character's voice with a real audio reference, so
the template declares `LoadAudio` slots next to its plate slots. The reference
node takes its autogrow inputs as dotted keys (`ref_audios.ref_audio_0`); the
ComfyUI API silently drops the nested form, which is how every earlier trailer
take shipped without a voice.
"""

from pathlib import Path

import pytest
from test_comfy import (
    JOB_ID,
    PROMPT_TEXT,
    _comfy,
    _document,
    _FakeComfy,
    _history,
    _lease,
    _packaged,
    _run,
    _template_directory,
)

from outbound_gpu_worker_pool import JobPayloadValue
from outbound_gpu_worker_pool.comfy import TemplateRegistry, input_schema
from outbound_gpu_worker_pool.plugins import PluginRequestRejected

DIALOGUE_CAPABILITY = "video.minimax_h3.dialogue.v1"
REFERENCE_NODE = "104"
VOICE_A_NODE = "301"
VOICE_B_NODE = "302"
VIDEO_NODE = "91"
OPENING_KEY = "inputs/pool/opening.png"
END_PLATE_KEY = "inputs/pool/end.png"
VOICE_KEY = "inputs/pool/voice.wav"
ALL_KEYS = (OPENING_KEY, END_PLATE_KEY, VOICE_KEY)


def _media(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for key in ALL_KEYS:
        path = tmp_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key.encode())
        paths[key] = path
    return paths


def _payload(**overrides: JobPayloadValue) -> dict[str, JobPayloadValue]:
    payload: dict[str, JobPayloadValue] = {
        "prompt": PROMPT_TEXT,
        "length": 277,
        "images": {"opening_frame": OPENING_KEY, "end_plate": END_PLATE_KEY},
        "audios": {"voice_a": VOICE_KEY},
    }
    payload.update(overrides)
    return payload


def test_the_template_publishes_plate_and_voice_slots() -> None:
    template = _packaged().template(DIALOGUE_CAPABILITY)
    assert template is not None
    assert [slot.name for slot in template.image_slots] == [
        "opening_frame",
        "end_plate",
    ]
    assert [slot.name for slot in template.audio_slots] == ["voice_a", "voice_b"]
    schema = input_schema(template)
    assert schema["properties"]["audios"] == {
        "type": "object",
        "additionalProperties": False,
        "properties": {"voice_a": {"type": "string"}, "voice_b": {"type": "string"}},
        "required": [],
    }
    # Plates are required, voices are not.
    assert "images" in schema["required"]
    assert "audios" not in schema["required"]
    reference = template.graph[REFERENCE_NODE]["inputs"]
    # <Picture 1> is the concrete opening frame, which the graph also pins at
    # frame 0; <Picture 2> is the plate the shot must land on, pinned at -1.
    assert reference["ref_images.ref_image_0"] == ["203", 0]
    assert reference["ref_images.ref_image_1"] == ["202", 0]
    assert template.graph["410"]["inputs"]["image"] == ["203", 0]
    assert template.graph["410"]["inputs"]["frame_idx"] == 0
    assert template.graph["411"]["inputs"]["image"] == ["202", 0]
    assert template.graph["411"]["inputs"]["frame_idx"] == -1
    assert reference["ref_audios.ref_audio_0"] == [VOICE_A_NODE, 0]
    assert reference["ref_audios.ref_audio_1"] == [VOICE_B_NODE, 0]
    assert template.model_version == "ref2va-int8"


def test_a_voice_slot_must_bind_a_load_audio_node(tmp_path: Path) -> None:
    document = _document(
        audio_slots=[{"name": "voice", "node_id": "2", "required": False}]
    )
    directory = _template_directory(tmp_path, ("workflow", document))

    with pytest.raises(ValueError, match="audio slot voice must bind a LoadAudio"):
        TemplateRegistry.from_directory(directory)


async def test_a_template_without_audio_slots_rejects_an_audios_key() -> None:
    async with _comfy(_FakeComfy()) as plugin:
        with pytest.raises(PluginRequestRejected, match="unsupported keys"):
            plugin.validate(_lease({"prompt": PROMPT_TEXT, "audios": {}}))


async def test_validate_rejects_a_voice_the_lease_did_not_grant() -> None:
    async with _comfy(_FakeComfy()) as plugin:
        with pytest.raises(PluginRequestRejected, match="voice_a must bind a granted"):
            plugin.validate(
                _lease(
                    _payload(),
                    input_keys=(OPENING_KEY, END_PLATE_KEY),
                    capability_id=DIALOGUE_CAPABILITY,
                )
            )


async def test_validate_rejects_a_granted_input_no_slot_binds() -> None:
    async with _comfy(_FakeComfy()) as plugin:
        with pytest.raises(PluginRequestRejected, match="no slot binds"):
            plugin.validate(
                _lease(
                    _payload(audios={}),
                    input_keys=ALL_KEYS,
                    capability_id=DIALOGUE_CAPABILITY,
                )
            )


async def test_validate_requires_every_plate() -> None:
    async with _comfy(_FakeComfy()) as plugin:
        with pytest.raises(
            PluginRequestRejected, match="image slot end_plate is required"
        ):
            plugin.validate(
                _lease(
                    _payload(images={"opening_frame": OPENING_KEY}),
                    input_keys=(OPENING_KEY, VOICE_KEY),
                    capability_id=DIALOGUE_CAPABILITY,
                )
            )


async def test_execute_uploads_the_voice_and_drops_the_unbound_one(
    tmp_path: Path,
) -> None:
    fake = _FakeComfy(histories=[_history(completed=True)], stored_as="stored")
    run = _run(tmp_path, input_paths=_media(tmp_path))

    async with _comfy(fake) as plugin:
        request = plugin.validate(
            _lease(_payload(), input_keys=ALL_KEYS, capability_id=DIALOGUE_CAPABILITY)
        )
        output = await plugin.execute(run.context, request)

    graph = fake.graphs[0]
    assert fake.uploaded_names == [
        f"{JOB_ID}-opening_frame.png",
        f"{JOB_ID}-end_plate.png",
        f"{JOB_ID}-voice_a.wav",
    ]
    assert graph[VOICE_A_NODE]["inputs"]["audio"] == "stored"
    reference = graph[REFERENCE_NODE]["inputs"]
    assert reference["ref_audios.ref_audio_0"] == [VOICE_A_NODE, 0]
    # The unbound voice leaves neither its node nor a dangling dotted link.
    assert VOICE_B_NODE not in graph
    assert "ref_audios.ref_audio_1" not in reference
    assert reference["length"] == 277
    assert graph[VIDEO_NODE]["inputs"]["fps"] == 24
    assert output.model_version == "ref2va-int8"
    assert output.content_type == "video/mp4"
