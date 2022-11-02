import os
import re
import shutil
from pathlib import Path
from typing import Dict, Generator, List, Pattern, Union

import pytest
from _pytest.config import Config
from packaging.version import VERSION_PATTERN


@pytest.fixture(name="root_dir", scope="session")
def fixture_root_dir(pytestconfig: Config) -> str:
    test_dir: str
    test_paths: Union[None, str, List[str]] = pytestconfig.inicfg.get("testpaths")
    if not test_paths:
        default_dir: str = "tests"
        if not pytestconfig.args:
            test_dir = default_dir
        else:
            first_arg: str = pytestconfig.args[0]
            test_dir = first_arg if os.path.isdir(first_arg) else default_dir
    elif isinstance(test_paths, list):
        test_dir = test_paths[0]
    else:
        assert isinstance(test_paths, str)
        test_dir = test_paths

    test_dir = os.path.join(str(pytestconfig.rootpath), test_dir)
    test_dir = os.path.abspath(test_dir)
    assert os.path.isdir(test_dir)
    return test_dir


@pytest.fixture(name="resources_dir", scope="session")
def fixture_resources_dir(root_dir: str) -> str:
    res_dir: str = os.path.join(root_dir, "resources")
    assert os.path.isdir(res_dir)
    return res_dir


@pytest.fixture(name="clean_dir")
def fixture_clean_dir(tmp_path: Path) -> Generator[None, None, None]:
    cwd_og = os.getcwd()
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    os.chdir(tmp_path)
    yield
    os.chdir(cwd_og)
    shutil.rmtree(tmp_path)


@pytest.fixture(name="env", autouse=True)
def fixture_env() -> Generator[None, None, None]:
    env_og: Dict[str, str] = os.environ.copy()

    yield

    os.environ.clear()
    os.environ.update(env_og)


@pytest.fixture(name="version_regex", scope="session")
def fixture_version_regex() -> Pattern:
    return re.compile(
        r"^\s*" + VERSION_PATTERN + r"\s*$", flags=re.VERBOSE | re.IGNORECASE
    )
