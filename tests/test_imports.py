import importlib

MODULES = [
    'scripts.load_data',
    'scripts.query_data',
    'scripts.visualize',
    'ai_config',
]


def test_imports():
    for m in MODULES:
        importlib.import_module(m)
