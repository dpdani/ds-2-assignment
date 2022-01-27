import contextlib
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path


@contextlib.contextmanager
def cd(path: Path):
    old = Path().absolute()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def compile(p: Path):
    with cd(p.parent):
        subprocess.check_call(
            f'xelatex.exe -synctex=1 -interaction=nonstopmode "{p.stem}"{p.suffix}'
        )


if __name__ == '__main__':
    with Pool(processes=None) as pool:
        pool.map(
            compile,
            (Path() / "report" / "figures").glob("**/*.tex")
        )
