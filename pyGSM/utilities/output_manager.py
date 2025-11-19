import os
import tempfile
import weakref
import functools
import h5py
import numpy as np

from .manage_xyz import write_xyz, format_xyz, format_molden

__all__ = [
    "OutputManager",
    "XYZWriter",
    "HDF5Checkpointer",
    "CombinedOutputSystem"
]

class OutputManager:

    def __init__(self, output_dir, **tempdir_opts):
        self.output_dir = output_dir
        self._temp_dir = None
        self._root = None
        self.tempdir_opts = tempdir_opts
        self._call_depth = 0

    _managers = weakref.WeakValueDictionary()
    @classmethod
    def lookup(cls, key):
        if key in cls._managers:
            manager = cls._managers[key]
        elif isinstance(key, OutputManager):
            manager = key
        else:
            manager = cls(key)

        return manager
    def register(self, key):
        self._managers[key] = self

    def __enter__(self):
        self._call_depth = max(self._call_depth, 0) + 1
        if self._call_depth == 1:
            if self.output_dir is not None:
                if self.output_dir is True:
                    self._temp_dir = tempfile.TemporaryDirectory()
                    self._temp_dir.__enter__()
                    self._root = self._temp_dir.name
                else:
                    if hasattr(self.output_dir, '__enter__'):
                        self.output_dir.__enter__()
                        self._root = self.output_dir.name
                    else:
                        os.makedirs(self.output_dir, exist_ok=True)
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._call_depth = max(self._call_depth-1, 0)
        if self._call_depth < 1:
            self._root = None
            if hasattr(self.output_dir, '__exit__'):
                self.output_dir.__exit__(exc_type, exc_val, exc_tb)
            elif hasattr(self._temp_dir, '__exit__'):
                self._temp_dir.__exit__(exc_type, exc_val, exc_tb)
                self._temp_dir = None

    def resolve_path(self, *path):
        if self._call_depth < 1:
            cls = type(self)
            raise ValueError(f"{cls.__name__} must be used as a context manager")
        if self._root is None:
            return None
        else:
            return os.path.join(self._root, *path)

    def write_output_file(self, path, *data, writer, _iherit_opts=None, **writer_opts):
        # this is so we can, e.g., directly write to a `.zip` file
        # if we need archives or to turn of writing at all if output is disabled
        if not hasattr(path, '__getitem__'):
            path = (path,)
        file = self.resolve_path(*path)
        if file is not None:
            if _iherit_opts is None:
                _iherit_opts = {}
            res = writer(file, *data, **dict(_iherit_opts, **writer_opts))
            if res is not None:
                return res
            else:
                return file
        else:
            return None

class XYZWriter:
    default_xyz_format = 'xyz'
    def __init__(self, output_manager:OutputManager, xyz_format=None):
        self.manager = output_manager
        self.format = xyz_format

    def __enter__(self):
        self.manager.__enter__()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.__exit__(exc_type, exc_val, exc_tb)

    xyz_formats = {
        'xyz':format_xyz,
        'molden':format_molden,
    }
    @classmethod
    def resolve_xyz_handler(cls, format):
        if isinstance(format, str):
            format = cls.xyz_formats[format]

        @functools.wraps(write_xyz)
        def write_file(file, atoms, geoms, *etc, **opts):
            return write_xyz(file,
                             atoms,
                             geoms,
                             *etc,
                             formatter=format,
                             **opts)
        return write_file
    def write_xyz(self, path, atoms, geoms, *other, format=None, **writer_opts):
        if format is None:
            format = self.format
        writer = self.resolve_xyz_handler(format)
        return self.manager.write_output_file(path, atoms, geoms, *other,
                                              writer=writer,
                                              _iherit_opts=writer_opts)

    def __call__(self, path, atoms, geoms, *other, **opts):
        return self.write_xyz(path, atoms, geoms, *other, **opts)

class HDF5FileInterface:
    def __init__(self, h5py_base:h5py.File):
        self.stream = h5py_base

    def set_item(self, key, value):
        if isinstance(value, dict):
            obj = type(self)(self.stream.create_group(key))
            for k,v in value.items():
                obj.set_item(k, v)
        else:
            self.stream[key] = value
    def __setitem__(self, key, value):
        self.set_item(key, value)


class HDF5Checkpointer:
    def __init__(self, checkpoint_file="checkpoint.hdf5", mode='a+b', **file_opts):
        self.checkpoint_file = checkpoint_file
        self.opts = dict(file_opts, mode=mode)
        self._was_open = False
        self._stream = None

    def __enter__(self):
        if self._stream is None:
            self._was_open = hasattr(self.checkpoint_file, 'write')
            if not self._was_open:
                self._stream = open(self.checkpoint_file, **self.opts).__enter__()
        return h5py.File(self._stream, "a")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._was_open:
            self._stream.__exit__(exc_type, exc_val, exc_tb)
        self._stream = None

class CombinedOutputSystem:
    def __init__(self,
                 output_dir,
                 scratch_dir=None,
                 xyz_format='xyz',
                 checkpoint_file="checkpoint.hdf5"
                 ):
        self.output_dir = OutputManager.lookup(output_dir)
        self.scratch_dir = OutputManager.lookup(scratch_dir)
        self.output_xyz_writer = XYZWriter(self.output_dir, xyz_format)
        self.scratch_xyz_writer = XYZWriter(self.scratch_dir, xyz_format)
        self.checkpointer = HDF5Checkpointer(self.output_dir, checkpoint_file)

    def __enter__(self):
        self.output_dir.__enter__()
        self.scratch_dir.__enter__()
        self.checkpointer.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.output_dir.__exit__(exc_type, exc_val, exc_tb)
        finally:
            try:
                self.scratch_dir.__exit__(exc_type, exc_val, exc_tb)
            finally:
                self.checkpointer.__exit__(exc_type, exc_val, exc_tb)