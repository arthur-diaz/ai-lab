from importlib import import_module


def test_package_is_importable():
    package = import_module("structured_outputs")

    assert package.__name__ == "structured_outputs"
