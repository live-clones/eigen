#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
#
# Content-addressed pass cache for CI test jobs (driven by
# test.linux.script.sh, run from inside the build directory).
#
# `plan` reads a `ctest --show-only=json-v1` dump and keys every enumerated
# test by an environment fingerprint plus the content of its definition:
# command elements that are files (the test binary, and the emulator for
# cross-compiled jobs) are replaced by a digest of their bytes, remaining
# elements and the CTest properties are taken literally with build-directory
# paths normalized away.  Tests whose key the manifest already records as a
# first-attempt pass go to --skip-out; every keyable test goes to --keys-out
# for the later `record` step.
#
# `record` folds the run's first-attempt results back into the manifest,
# using the dashboard run's Test.xml as the authoritative status source:
# only tests with Status="passed" are recorded ("failed" and "notrun" cover
# failures, timeouts, missing executables, and SKIP_RETURN_CODE skips).
# New passes are appended, entries that proved useful again (skipped or
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
import xml.etree.ElementTree as ElementTree


def file_digest(path, _cache={}):
    digest = _cache.get(path)
    if digest is None:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = _cache[path] = h.hexdigest()
    return digest


def normalize(text, builddirs):
    """Strips build-directory prefixes so the key does not depend on where
    the checkout happens to live."""
    for builddir in builddirs:
        text = text.replace(builddir + os.sep, "").replace(builddir, ".")
    return text


def test_key(test, fingerprint, builddirs):
    """Digest of the fingerprint and the test's definition, or None if the
    command does not end in an executable under the build directory."""
    command = test.get("command")
    if not command:
        return None
    last = os.path.realpath(command[-1])
    if not (last.startswith(builddirs[0] + os.sep) and os.path.isfile(last) and os.access(last, os.X_OK)):
        return None
    h = hashlib.sha256()
    h.update(fingerprint.encode() + b"\0")
    for element in command:
        path = os.path.realpath(element)
        if os.path.isfile(path):
            h.update(b"file:" + file_digest(path).encode() + b"\0")
        else:
            h.update(b"arg:" + normalize(element, builddirs).encode() + b"\0")
    properties = sorted(test.get("properties", []), key=lambda p: str(p.get("name")))
    h.update(b"props:" + normalize(json.dumps(properties, sort_keys=True), builddirs).encode())
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
    builddirs = [os.path.realpath(os.getcwd())]
    if os.getcwd() != builddirs[0]:
        builddirs.append(os.getcwd())
    keys = []
    skip = []
    unkeyed = 0
    for test in tests:
        key = test_key(test, args.fingerprint, builddirs)
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
    # The dashboard <Test> elements carry a Status attribute ("passed",
    # "failed", or "notrun"); the bare <Test> entries in <TestList> do not
    # and are ignored.  A missing or unparsable Test.xml means ctest died
    # without completing the test phase: record nothing rather than trust it.
    try:
        root = ElementTree.parse(args.test_xml).getroot()
    except (OSError, ElementTree.ParseError) as error:
        print("test cache record: cannot read %s (%s); recording nothing" % (args.test_xml, error))
        return
    ran = set()
    passed = set()
    for test in root.iter("Test"):
        status = test.get("Status")
        name = test.find("Name")
        if status is None or name is None:
            continue
        ran.add(name.text)
        if status == "passed":
            passed.add(name.text)
    manifest = read_manifest(args.manifest)
    new = refreshed = 0
    for name, key in keys.items():
        if name in passed:
            if key in manifest:
                del manifest[key]
                refreshed += 1
            else:
                new += 1
            manifest[key] = "%s %s" % (key, name)
        elif name in skipped and name not in ran and key in manifest:
            # Skipped this run on the strength of its manifest entry; move
            # the entry to the back so trimming ages out unused ones first.
            del manifest[key]
            manifest[key] = "%s %s" % (key, name)
            refreshed += 1
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
    p.add_argument("--test-xml", required=True)
    p.add_argument("--max-entries", type=int, default=20000)
    p.set_defaults(func=record)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
