"""Offline tests for the image seam."""

from storyillus.imagegen.fake import FakeImage


def test_the_same_seed_gives_the_same_image():
    backend = FakeImage()
    first = backend.generate("a lab at night", seed=42)
    second = backend.generate("a different prompt", seed=42)
    assert first.getpixel((0, 0)) == second.getpixel((0, 0))


def test_a_different_seed_gives_a_different_image():
    backend = FakeImage()
    assert backend.generate("x", seed=1).getpixel((0, 0)) != backend.generate(
        "x", seed=2
    ).getpixel((0, 0))


def test_calls_are_recorded_for_assertions():
    backend = FakeImage()
    backend.generate("a lab", negative="blurry", seed=3)
    assert backend.calls == [("a lab", "blurry", 3)]


def test_size_is_configurable():
    assert FakeImage(size=(32, 16)).generate("x").size == (32, 16)
