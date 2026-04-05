import sys

from main import main


if __name__ == "__main__":
    if "--trainer" not in sys.argv:
        sys.argv.extend(["--trainer", "improved"])
    main()
