#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
#
# Content-addressed pass cache for CI test jobs (driven by
# test.linux.script.sh, run from inside the build directory).
#
# `plan` reads a `ctest --show-only=json-v1` dump and keys every enumerated
# test by an environment fingerprint plus the content of its command line:
# elements that are files (the test binary, and the emulator for
# cross-compiled jobs) are replaced by a digest of their bytes, everything
# else is taken literally.  Tests whose key the manifest already records as
# a first-attempt pass go to --skip-out; every keyable test goes to
# --keys-out for the later `record` step.
#
# `record` folds the run's first-attempt results back into the manifest:
# new passes are appended, entries that proved useful again (skipped or
# re-passed) move to the end, and the file is trimmed to --max-entries, so
# it ages out roughly least-recently-useful first.
#
# A test is keyable only if its command ends in an executable inside the
# build directory.  Anything else (e.g. a test that invokes the compiler on
# tree sources) has an outcome that depends on more than the artifact's
# content and must always run.

import argparse
import hashlib
import json
import os


def file_digest(path, _cache={}):
    digest = _cache.get(path)
    if digest is None:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = _cache[path] = h.hexdigest()
    return digest


def test_key(command, fingerprint, builddir):
    """Digest of the fingerprint and the command's content, or None if the
    command does not end in an executable under the build directory."""
    last = os.path.realpath(command[-1])
    if not (last.startswith(builddir + os.sep) and os.path.isfile(last) and os.access(last, os.X_OK)):
        return None
    h = hashlib.sha256()
    h.update(fingerprint.encode() + b"\0")
    for element in command:
        path = os.path.realpath(element)
        if os.path.isfile(path):
            h.update(b"file:" + file_digest(path).encode() + b"\0")
        else:
            h.update(b"arg:" + element.encode() + b"\0")
    return h.hexdigest()


def read_manifest(path):
    """The manifest as an ordered {key: "key testname"} dict."""
    entries = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                fields = line.split()
                if fields:
                    entries[fields[0]] = line.rstrip("\n")
    return entries


def write_manifest(path, entries, max_entries):
    lines = list(entries.values())[-max_entries:]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("".join(line + "\n" for line in lines))
    os.replace(tmp, path)


def plan(args):
    with open(args.tests_json) as f:
        tests = json.load(f)["tests"]
    manifest = read_manifest(args.manifest)
    builddir = os.path.realpath(os.getcwd())
    keys = []
    skip = []
    unkeyed = 0
    for test in tests:
        command = test.get("command")
        key = test_key(command, args.fingerprint, builddir) if command else None
        if key is None:
            unkeyed += 1
            continue
        keys.append((test["name"], key))
        if key in manifest:
            skip.append(test["name"])
    with open(args.keys_out, "w") as f:
        f.writelines("%s %s\n" % (key, name) for name, key in keys)
    with open(args.skip_out, "w") as f:
        f.writelines(name + "\n" for name in skip)
    print(
        "test cache plan: %d tests selected, %d cached passes to skip, %d to run (%d not keyable)"
        % (len(tests), len(skip), len(tests) - len(skip), unkeyed)
    )


def record(args):
    keys = {}
    with open(args.keys) as f:
        for line in f:
            fields = line.split()
            if len(fields) == 2:
                keys[fields[1]] = fields[0]
    with open(args.skip) as f:
        skipped = {line.strip() for line in f if line.strip()}
    # LastTestsFailed.log lines are "<index>:<testname>"; timeouts and
    # not-run tests are listed too, which is exactly what must not be
    # recorded as a pass.
    failed = set()
    with open(args.failed) as f:
        for line in f:
            line = line.strip()
            if line:
                failed.add(line.split(":", 1)[-1])
    manifest = read_manifest(args.manifest)
    new = refreshed = 0
    for name, key in keys.items():
        if name in failed:
            continue
        if key in manifest:
            del manifest[key]
            manifest[key] = "%s %s" % (key, name)
            refreshed += 1
        elif name not in skipped:
            manifest[key] = "%s %s" % (key, name)
            new += 1
    write_manifest(args.manifest, manifest, args.max_entries)
    print(
        "test cache record: %d new first-attempt passes, %d entries refreshed, %d entries total"
        % (new, refreshed, len(manifest))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--tests-json", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--fingerprint", required=True)
    p.add_argument("--skip-out", required=True)
    p.add_argument("--keys-out", required=True)
    p.set_defaults(func=plan)

    p = sub.add_parser("record")
    p.add_argument("--manifest", required=True)
    p.add_argument("--keys", required=True)
    p.add_argument("--skip", required=True)
    p.add_argument("--failed", required=True)
    p.add_argument("--max-entries", type=int, default=20000)
    p.set_defaults(func=record)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
