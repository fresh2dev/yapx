from typing import List, Pattern
from unittest import mock

from _pytest.capture import CaptureFixture, CaptureResult

from yapx.__main__ import main
from yapx.__main__ import sys as target_sys


def test_main_version(capsys: CaptureFixture, version_regex: Pattern):
    # 1. ARRANGE
    args: List[str] = ["version"]

    # 2. ACT
    with mock.patch.object(target_sys, "argv", [""] + args):
        main()

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert not captured.err
    assert captured.out
    assert version_regex.search(captured.out), "invalid version"
