#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fail fast when a Ray temp root cannot fit Linux AF_UNIX sockets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

AF_UNIX_PATH_LIMIT = 107
# Representative Ray 2.x session component, including a seven-digit PID.
RAY_SOCKET_SUFFIX = (
    "ray/session_2026-08-10_18-04-31_123456_1234567/sockets/plasma_store"
)


def representative_socket_path(tmpdir: str | os.PathLike[str]) -> Path:
    return Path(tmpdir).expanduser().resolve() / RAY_SOCKET_SUFFIX


def validate(tmpdir: str | os.PathLike[str]) -> Path:
    socket_path = representative_socket_path(tmpdir)
    byte_length = len(os.fsencode(socket_path))
    if byte_length > AF_UNIX_PATH_LIMIT:
        raise ValueError(
            "RAY_TMPDIR is too long for Ray's AF_UNIX plasma-store socket: "
            f"estimated {byte_length} bytes > {AF_UNIX_PATH_LIMIT}: {socket_path}"
        )
    return socket_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tmpdir")
    args = parser.parse_args()
    socket_path = validate(args.tmpdir)
    print(
        "RAY_TMPDIR_PATH_CHECK_PASS "
        f"bytes={len(os.fsencode(socket_path))}/{AF_UNIX_PATH_LIMIT} "
        f"socket={socket_path}"
    )


if __name__ == "__main__":
    main()
