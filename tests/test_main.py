import sys

from src.main import main


def test_main():
    assert main() == 0, "Should be 0"  # nosec: B101


def test_python_version():
    assert sys.version_info >= (  # nosec: B101
        3,
        8,
    ), "Python version should be >= 3.8"


if __name__ == "__main__":
    test_main()
    test_python_version()
    print("Everything passed")
