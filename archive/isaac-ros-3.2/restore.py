#!/usr/bin/env python3
"""Verify or restore the frozen Isaac snapshot; never overwrite existing roots."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile

BASE = Path(__file__).resolve().parent


def sha256(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def safe_path(value):
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Unsafe manifest path: ' + value)
    return Path(*p.parts)


def check_file(p, entry):
    kind = entry['type']
    if kind == 'directory':
        if p.exists() and (p.is_symlink() or not p.is_dir()):
            raise ValueError('Invalid directory: ' + str(p))
        return
    if kind == 'symlink':
        if not p.is_symlink() or os.readlink(p) != entry['target']:
            raise ValueError('Symlink mismatch: ' + str(p))
        return
    if kind != 'file' or not p.is_file() or p.is_symlink():
        raise ValueError('Missing or invalid file: ' + str(p))
    with p.open('rb') as stream:
        if p.stat().st_size != entry['bytes'] or sha256(stream) != entry['sha256']:
            raise ValueError('Content mismatch: ' + str(p))


def verify_archive(manifest):
    count = 0
    for root in manifest['roots']:
        snapshot = BASE / safe_path(root['snapshot'])
        metadata = [e for e in root['entries'] if safe_path(e['path']).parts[0] == '.git']
        ordinary = [e for e in root['entries'] if e not in metadata]
        expected = {e.get('stored_path', e['path']) for e in ordinary if e['type'] != 'directory'}
        actual = {str(p.relative_to(snapshot)) for p in snapshot.rglob('*')
                  if p.is_symlink() or not p.is_dir()}
        if actual != expected:
            raise ValueError('Snapshot file list mismatch: ' + str(snapshot))
        for entry in ordinary:
            check_file(snapshot / safe_path(entry.get('stored_path', entry['path'])), entry)
        if metadata:
            with tarfile.open(BASE / safe_path(root['git_metadata']), 'r:gz') as tf:
                members = tf.getmembers()
                names = [m.name.rstrip('/') for m in members]
                if len(set(names)) != len(names) or set(names) != {e['path'] for e in metadata}:
                    raise ValueError('Git metadata member list mismatch')
                for entry in metadata:
                    member = tf.getmember(entry['path'])
                    safe_path(member.name)
                    if entry['type'] == 'file':
                        if not member.isfile() or member.size != entry['bytes']:
                            raise ValueError('Git metadata type/size mismatch')
                        with tf.extractfile(member) as stream:
                            if sha256(stream) != entry['sha256']:
                                raise ValueError('Git metadata content mismatch')
                    elif entry['type'] == 'directory':
                        if not member.isdir():
                            raise ValueError('Git directory mismatch')
                    elif not member.issym() or member.linkname != entry['target']:
                        raise ValueError('Git symlink mismatch')
        count += sum(e['type'] == 'file' for e in root['entries'])
    print('Verified archive:', count, 'files plus symlinks and directory metadata')


def verify_existing(manifest, destination):
    for root in manifest['roots']:
        target = destination / Path(root['source']).name
        if target.is_symlink() or not target.is_dir():
            raise ValueError('Missing or invalid root: ' + str(target))
        expected = {e['path'] for e in root['entries']}
        actual = {str(p.relative_to(target)) for p in target.rglob('*')}
        if actual != expected:
            raise ValueError('Local file list changed: ' + str(target))
        for entry in root['entries']:
            p = target / safe_path(entry['path'])
            check_file(p, entry)
            if stat.S_IMODE(p.lstat().st_mode) != entry['mode']:
                raise ValueError('Local permissions changed: ' + str(p))
    print('Verified existing directories:', destination)


def restore(manifest, destination):
    destination = destination.expanduser().resolve()
    targets = [destination / Path(r['source']).name for r in manifest['roots']]
    if any(p.exists() or p.is_symlink() for p in targets):
        raise ValueError('Refusing to overwrite an existing destination')
    destination.mkdir(parents=True, exist_ok=True)
    for root, target in zip(manifest['roots'], targets):
        target.mkdir()
        metadata = tarfile.open(BASE / safe_path(root['git_metadata']), 'r:gz') if root.get('git_metadata') else None
        try:
            for entry in root['entries']:
                rel = safe_path(entry['path'])
                p = target / rel
                if entry['type'] == 'directory':
                    p.mkdir(parents=True, exist_ok=True)
            # Write regular files before creating any symlinks.
            for entry in root['entries']:
                if entry['type'] != 'file':
                    continue
                rel = safe_path(entry['path'])
                p = target / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                if rel.parts[0] == '.git':
                    with metadata.extractfile(entry['path']) as src, p.open('xb') as out:
                        shutil.copyfileobj(src, out)
                else:
                    shutil.copyfile(BASE / root['snapshot'] / safe_path(entry.get('stored_path', entry['path'])), p)
                p.chmod(entry['mode'])
            for entry in root['entries']:
                if entry['type'] == 'symlink':
                    p = target / safe_path(entry['path'])
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.symlink_to(entry['target'])
            for entry in reversed(root['entries']):
                if entry['type'] == 'directory':
                    (target / safe_path(entry['path'])).chmod(entry['mode'])
        finally:
            if metadata:
                metadata.close()
    verify_existing(manifest, destination)
    print('Restored all three original directory names')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--verify-only', action='store_true')
    modes.add_argument('--destination', type=Path)
    modes.add_argument('--check-existing', type=Path)
    args = parser.parse_args()
    manifest = json.loads((BASE / 'MANIFEST.json').read_text())
    verify_archive(manifest)
    if args.check_existing:
        verify_existing(manifest, args.check_existing)
    elif args.destination:
        restore(manifest, args.destination)
