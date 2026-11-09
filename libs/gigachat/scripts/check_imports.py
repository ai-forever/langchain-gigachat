import sys
import traceback
from importlib import import_module
from pathlib import Path

if __name__ == "__main__":
    source_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source_root))
    files = sys.argv[1:]
    has_failure = False
    for file in files:
        try:
            module_path = Path(file).resolve().relative_to(source_root).with_suffix("")
            parts = list(module_path.parts)
            if parts[-1] == "__init__":
                parts.pop()
            import_module(".".join(parts))
        except Exception:
            has_failure = True
            print(file)  # noqa: T201
            traceback.print_exc()
            print()  # noqa: T201

    sys.exit(1 if has_failure else 0)
