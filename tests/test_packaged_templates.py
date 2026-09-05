"""Every template shipped in the package must load and validate.

An enrolled machine's agent reads these from its templates directory; a malformed
template would only surface as a startup failure on the box.
"""

from outbound_gpu_worker_pool.comfy import (
    PACKAGED_TEMPLATES_DIRECTORY,
    TemplateRegistry,
)


def test_every_packaged_template_loads_and_validates() -> None:
    registry = TemplateRegistry.from_directory(PACKAGED_TEMPLATES_DIRECTORY)
    capability_ids = {template.capability_id for template in registry.templates}
    assert {
        "video.minimax_h3.text_to_video.v1",
        "image.flux2_klein.subject.v1",
        "image.flux2_klein.subject.v2",
    } <= capability_ids


def test_subject_template_exposes_only_prompt_steps_and_seed() -> None:
    registry = TemplateRegistry.from_directory(PACKAGED_TEMPLATES_DIRECTORY)
    template = registry.template("image.flux2_klein.subject.v1")
    assert template is not None
    assert {entry.name for entry in template.inputs} == {"prompt", "steps", "seed"}
    assert template.image_slots == ()
    assert template.output_content_type == "image/png"


def test_subject_v2_template_adds_three_optional_reference_slots() -> None:
    registry = TemplateRegistry.from_directory(PACKAGED_TEMPLATES_DIRECTORY)
    template = registry.template("image.flux2_klein.subject.v2")
    assert template is not None
    assert {entry.name for entry in template.inputs} == {"prompt", "steps", "seed"}
    assert [slot.name for slot in template.image_slots] == ["ref_1", "ref_2", "ref_3"]
    assert all(not slot.required for slot in template.image_slots)
    assert template.output_content_type == "image/png"
