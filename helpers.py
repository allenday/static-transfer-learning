import hashlib


def get_sha1_hash(*values):
    sha = hashlib.sha1()

    for value in values:
        sha.update(value)

    return sha.hexdigest()
