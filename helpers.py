import hashlib
from _hashlib import HASH as Hash
from pathlib import Path
from typing import Union


def get_sha1_hash(*values):
    sha = hashlib.sha1()

    for value in values:
        if type(value) != bytes:
            value = value.encode('utf8')
        sha.update(value)

    return sha.hexdigest()


def get_sha1_from_file(filename: Union[str, Path], hash: Hash) -> Hash:
    assert Path(filename).is_file()
    with open(str(filename), "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash.update(chunk)
    return hash


def get_sha1_hash_file(filename: Union[str, Path], hash: Hash) -> str:
    return str(get_sha1_from_file(filename, hash).hexdigest())


def get_sha1_hash_from_dir(directory: Union[str, Path], hash: Hash = None) -> str:
    assert Path(directory).is_dir()
    if hash is None:
        hash = hashlib.sha1()
    for path in sorted(Path(directory).iterdir()):
        hash.update(path.name.encode())
        if path.is_file():
            hash = get_sha1_from_file(path, hash)
        elif path.is_dir():
            hash = get_sha1_hash_from_dir(path, hash)
    return hash.hexdigest()
