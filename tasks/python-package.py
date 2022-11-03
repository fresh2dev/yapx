#!/usr/bin/env python3

import os
from typing import Any, Dict, List, Optional

import myke

PYENV_VERSION_FILE: str = ".python-base-version"
PYENV_VENV_FILE: str = ".python-version"

VERSION_FILE: str = "VERSION"
REPORT_DIR: str = "public"


@myke.cache()
def _get_package_root() -> str:
    return str(
        myke.read.cfg("setup.cfg").get("options.packages.find", {}).get("where", "src")
    )


@myke.cache()
def _get_package_dirs() -> List[str]:
    return myke.sh_stdout_lines(
        f"""
    find "{_get_package_root()}" \
        -mindepth 2 -maxdepth 2 -name "__init__.py" -exec dirname {{}} \\;
    """
    )


@myke.cache()
def _get_project_name() -> str:
    return myke.sh_stdout("python setup.py --name")


@myke.task
def x_get_version(_echo: bool = True) -> str:
    value: str = myke.sh_stdout("python setup.py --version")
    if _echo:
        print(value)
    return value


def _assert_unpublished(
    repository: Optional[str] = None,
    version: Optional[str] = None,
) -> List[str]:
    vers_published: List[str] = x_get_versions_published(
        repository=repository, _echo=False
    )

    if not version:
        version = x_get_version(_echo=False)

    if version in vers_published:
        raise SystemExit(
            ValueError(f"Version {version} already published; increment to continue.")
        )

    return vers_published


def _get_next_version(
    repository: Optional[str] = None,
    version: Optional[str] = None,
) -> str:
    if not version:
        version = os.getenv("CI_COMMIT_TAG", None)
        if not version:
            if os.path.exists(VERSION_FILE):
                version = myke.read.text(VERSION_FILE)
            else:
                version = "0.0.1a1"

    vers_published: List[str] = _assert_unpublished(
        repository=repository, version=version
    )

    # if in CI, but no tag, append next '.dev#' to version
    if os.getenv("CI") and not os.getenv("CI_COMMIT_TAG"):
        i: int = 0
        ver_suffix: str = f"dev{i}"
        next_dev_version: str = f"{version}.{ver_suffix}"
        while next_dev_version in vers_published:
            i += 1
            ver_suffix = f"dev{i}"
            next_dev_version = f"{version}.{ver_suffix}"
        version = next_dev_version

    if not os.path.exists(VERSION_FILE):
        x_set_version(version)

    return version


@myke.task
def x_set_version(
    version=myke.arg(None, pos=True), repository: Optional[str] = None
) -> None:
    version_og: Optional[str] = None
    try:
        version_og = x_get_version(_echo=False)
    except myke.exceptions.CalledProcessError:
        pass

    if not version:
        version = _get_next_version(version=version, repository=repository)

    if version_og != version:
        print(f"{version_og} --> {version}")
        myke.write.text(path=VERSION_FILE, content=version + os.linesep, overwrite=True)

        if not os.path.exists("MANIFEST.in"):
            myke.write.text(path="MANIFEST.in", content=f"include {VERSION_FILE}")

        myke.sh(
            r"sed 's/^version.*/version = file: VERSION/' setup.cfg > setup.cfg.tmp "
            r"&& mv setup.cfg.tmp setup.cfg"
        )

    assert version == x_get_version(_echo=False)

    for pkg in _get_package_dirs():
        myke.sh(f'cp "{VERSION_FILE}" "{os.path.join(pkg, VERSION_FILE)}"')


@myke.task_sh
def x_clean():
    return r"""
rm -rf dist build public src/*.egg-info .mypy_cache .pytest_cache .coverage .hypothesis .tox
find . -type d -name "__pycache__" | xargs -r rm -rf
"""


@myke.task
def x_env(
    version=None,
    name=None,
    extras: Optional[List[str]] = None,
    quiet=False,
):
    x_clean()

    if version:
        myke.sh(f"pyenv local {version}")
    elif os.path.exists(PYENV_VERSION_FILE):
        os.rename(PYENV_VERSION_FILE, PYENV_VENV_FILE)

    if not name:
        name = _get_project_name()

    myke.sh(
        f"""
export PYENV_VIRTUALENV_DISABLE_PROMPT=1 \\
&& pyenv virtualenv-delete -f {name} \\
&& pyenv virtualenv {name} \\
&& mv {PYENV_VENV_FILE} {PYENV_VERSION_FILE} \\
&& pyenv local {name}
"""
    )

    x_requirements(extras=extras, quiet=quiet)


@myke.task
def x_requirements(
    extras: Optional[List[str]] = None,
    quiet: bool = False,
) -> None:
    setup_cfg: Dict[str, Any] = myke.read.cfg("setup.cfg")

    install_requires: List[str] = [
        x
        for x in setup_cfg.get("options", {}).get("install_requires", "").splitlines()
        if x
    ]

    extras_require: Optional[Dict[str, str]] = setup_cfg.get("options.extras_require")

    if extras:
        if not extras_require:
            raise ValueError("Missing value for 'options.extras_require'")
        for e in extras:
            if e not in extras_require:
                raise ValueError(
                    f"Extra requirement '{e}' not one of: "
                    f"{','.join(extras_require.keys())}"
                )

    extra_reqs: List[str] = [
        req
        for grp, reqs in extras_require.items()
        if (not extras or grp in extras)
        for req in reqs.strip().splitlines()
        if req
    ]

    quiet_flag: str = ""
    if quiet:
        quiet_flag = "-q"

    build_reqs = ["pip", "setuptools", "wheel"]

    for reqs in build_reqs, install_requires, extra_reqs:
        if reqs:
            reqs = [x.replace("'", '"') for x in reqs]
            reqs_str: str = "'" + "' '".join(reqs) + "'"
            myke.sh(f"pip install {quiet_flag} {reqs_str}")


@myke.task
def x_get_versions_published(
    repository: Optional[str] = None, name: Optional[str] = None, _echo: bool = True
) -> List[str]:
    if not name:
        name = _get_project_name()

    pip_args: str = ""
    if repository and repository != "pypi":
        pypirc: str = os.path.join(os.path.expanduser("~"), ".pypirc")
        repo_conf: Dict[str, Dict[str, str]] = myke.read.cfg(pypirc)
        if repository not in repo_conf:
            raise ValueError(f"Specified repo '{repository}' not found in '{pypirc}'")
        repo_key: str = "repository"
        repo_url: Optional[str] = repo_conf.pop(repository).get(repo_key)
        if not repo_url:
            raise ValueError(
                f"Specified repo '{repository}' has no property '{repo_key}'"
            )

        from urllib.parse import ParseResult, urlparse

        url_parts: ParseResult = urlparse(repo_url)
        pip_args = (
            f"--trusted-host '{url_parts.netloc}' --index-url '{repo_url}/simple'"
        )

    values: List[str] = myke.sh_stdout_lines(
        f"pip install --disable-pip-version-check {pip_args} {name}== 2>&1"
        r" | tr ' ' '\n' | tr -d ',' | tr -d ')' | grep '^v\?[[:digit:]]'"
        r" || true"
    )

    if _echo:
        myke.print.lines(values)

    return values


@myke.task_sh
def x_format() -> str:
    return f"""
python -m isort {_get_package_root()} ./tests
python -m black {_get_package_root()} ./tests

if [ -e ~/.pre-commit-config.yml ]; then
    pre-commit run --config ~/.pre-commit-config.yml --all-files
fi
"""


@myke.task_sh
def x_check() -> str:
    dirs: List[str] = [_get_package_root(), "./tests"]
    joined_dirs = " ".join(dirs)

    return f"""
python -m flake8 {joined_dirs} || true
python -m pylint -f colorized {joined_dirs} || true
python -m mypy --install-types --non-interactive --html-report public/coverage-types || true
"""


@myke.task
def x_test(reports: bool = False) -> None:
    report_args: str = ""

    if reports:
        report_args = f""" \\
        --cov-report xml:{REPORT_DIR}/coverage.xml \\
        --cov-report html:{REPORT_DIR}/coverage \\
        --html {REPORT_DIR}/tests/index.html
        """

    myke.sh(
        f"""
PYTHONPATH={_get_package_root()} pytest --cov=src --cov-report term {report_args}
"""
    )


def _get_pyenv_versions() -> List[str]:
    return [
        x for x in myke.sh_stdout_lines("pyenv versions") if myke.utils.is_version(x)
    ]


@myke.task
def x_test_tox() -> None:
    env: Dict[str, str] = os.environ.copy()
    cur_ver: str = myke.sh_stdout("pyenv version-name")
    env["PYENV_VERSION"] = ":".join([cur_ver] + _get_pyenv_versions())
    myke.sh("tox", env=env)


@myke.task
def x_test_py() -> None:
    og_venv: Optional[str] = None
    if os.path.exists(PYENV_VERSION_FILE):
        og_venv = myke.read.text(PYENV_VERSION_FILE)

    proj_name: str = _get_project_name()

    try:
        for ver in _get_pyenv_versions():
            test_env_name = f"test-{proj_name}"
            x_env(ver, name=test_env_name, extras=["tests"], quiet=True)
            x_test()
            myke.sh(f"pyenv virtualenv-delete -f {test_env_name}")
    finally:
        myke.sh(f"pyenv local {proj_name}")
        if og_venv:
            myke.write.text(
                path=PYENV_VERSION_FILE, content=og_venv + os.linesep, overwrite=True
            )


@myke.task_sh
def x_mkdocs_serve() -> str:
    return "mkdocs serve --config-file config/mkdocs.yml"


@myke.task
def x_docs() -> None:

    myke.sh(
        f"""
python -m pdoc -o {REPORT_DIR}/srcdocs --docformat google --search --show-source {_get_package_root()}
"""
    )

    myke.sh(r"mkdocs build --clean --config-file config/mkdocs.yml")


#     return f"""
# echo "<h1>${{DRONE_REPO_NAME}}</h1>" > {REPORT_DIR}/index.html
# echo "<h2>Reports:</h2>" >> {REPORT_DIR}/index.html
# echo '<ul style="font-size: 1.5em">' >> {REPORT_DIR}/index.html
# find {REPORT_DIR} -mindepth 1 -maxdepth 1 -type d | sort | while read -r dir; do \
#     BASE_DIR="$(basename "${{dir}}")" \
#     && echo "<li><a href='${{BASE_DIR}}/' target='_blank'>${{BASE_DIR}}</a></li>" >> {REPORT_DIR}/index.html; \
#   done
# echo '</ul>' >> {REPORT_DIR}/index.html
# echo "<p>Ref: ${{DRONE_COMMIT}}</p>" >> {REPORT_DIR}/index.html
# echo "<p>Timestamp: ${{DRONE_BUILD_STARTED}}</p>" >> {REPORT_DIR}/index.html
# """


@myke.task
def x_reports() -> None:
    x_clean()
    x_check()
    x_test(reports=True)
    x_docs()


@myke.task_sh
def x_build() -> str:
    return r"""
python -m build
python -m twine check --strict dist/*
pip install --force-reinstall dist/*.whl
"""


@myke.task_sh
def publish(repository: str = "testpypi", build: bool = False) -> str:
    if build:
        x_build()

    return f"""
python -m twine upload --verbose --non-interactive -r {repository} dist/*
"""


@myke.task
def x_init() -> None:
    x_set_version()
    x_env()
    x_reports()


if __name__ == "__main__":
    myke.main(__file__)
