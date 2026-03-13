from langchain_gigachat import __all__, __version__

EXPECTED_ALL = ["GigaChat", "GigaChatEmbeddings"]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)


def test_package_version_available() -> None:
    assert __version__
