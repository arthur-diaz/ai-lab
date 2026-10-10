from importlib import import_module


def test_package_is_importable():
    package = import_module("embeddings_search")

    assert package.__name__ == "embeddings_search"
