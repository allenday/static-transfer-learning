import hashlib


def get_sha1_hash(*values):
    sha = hashlib.sha1()

    for value in values:
        if type(value) != bytes:
            value = value.encode('utf8')
        sha.update(value)

    return sha.hexdigest()
