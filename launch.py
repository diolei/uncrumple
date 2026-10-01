"""Detached training launcher: double-forks so the training run
survives the caller's shell and inherits no pipes. Usage:
./.venv/bin/python launch.py [--epochs N] [--stages S]
Log: /tmp/opencode/train-full.log ; artifact: uncrease/dist/."""

import os
import sys

LOG = "/tmp/opencode/train-full.log"


def main() -> None:
    args = sys.argv[1:]
    pid = os.fork()
    if pid > 0:
        print(f"launched training pid={pid} log={LOG}")
        return
    os.setsid()
    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)
    fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    py = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv/bin/python")
    os.execv(py, [py, "-m", "train.run", *args])


if __name__ == "__main__":
    main()
