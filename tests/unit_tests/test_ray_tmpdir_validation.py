# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests for Ray's Linux AF_UNIX path budget."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[2]
    / "experiments"
    / "flow_credit"
    / "analysis"
    / "validate_ray_tmpdir.py"
)
SPEC = importlib.util.spec_from_file_location("validate_ray_tmpdir", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_short_server_ray_tmpdirs_fit_socket_budget() -> None:
    for tmpdir in (
        "/mnt/models/gzw/raytmp/e0",
        "/mnt/models/gzw/raytmp/e1",
        "/mnt/models/gzw/raytmp/sweep",
        "/mnt/models/gzw/raytmp/strength",
        "/mnt/models/gzw/raytmp/fullppo",
    ):
        socket_path = MODULE.validate(tmpdir)
        assert len(str(socket_path).encode()) <= MODULE.AF_UNIX_PATH_LIMIT


def test_long_ray_tmpdir_is_rejected_before_ray_starts(tmp_path: Path) -> None:
    long_tmpdir = tmp_path / ("long-ray-directory-" * 8)

    with pytest.raises(ValueError, match="RAY_TMPDIR is too long"):
        MODULE.validate(long_tmpdir)
