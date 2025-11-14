
import pathlib
import json
import tempfile as tf

from . import Options as opts_handler

__all__ = [
    "is_filepath_like",
    "safe_open",
    "write_file",
    "read_file",
    "read_json",
    "write_json"
]

bad_file_chars = {" ", "\t", "\n"}
def is_filepath_like(file, bad_chars=None):
    if bad_chars is None:
        bad_chars = bad_file_chars
    return isinstance(file, pathlib.Path) or (
        isinstance(file, str)
        and all(b not in file for b in bad_chars)
    )

class safe_open:
    def __init__(self, file, **opts):
        self.file = file
        self._stream = None
        self.opts = opts
    def __enter__(self):
        if hasattr(self.file, 'seek'):
            return self.file
        else:
            self._stream = open(self.file, **self.opts)
            return self._stream.__enter__()
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stream is not None:
            return self._stream.__exit__(exc_type, exc_val, exc_tb)

class open_opts:
    mode: 'str'
    buffering: 'int'
    encoding: 'str | None'
    errors: 'str | None'
    newline: 'str | None'
    closefd: 'bool'
    opener: '(str, int)'

def write_file(file, data, mode='w+', **opts):
    with safe_open(file, mode=mode, **opts) as fs:
        fs.write(data)
    return file

def read_file(file, **opts):
    with safe_open(file, **opts) as fs:
        return fs.read()

def read_json(file, **opts):
    opts, js_opts = opts_handler.OptionsSet(opts).split(open_opts)
    with safe_open(file, **opts) as fs:
        return json.load(fs, **js_opts)

def write_json(file, data, mode="w+", **opts):
    opts, js_opts = opts_handler.OptionsSet(opts).split(open_opts)
    with safe_open(file, mode=mode, **opts) as fs:
        return json.dump(data, fs)