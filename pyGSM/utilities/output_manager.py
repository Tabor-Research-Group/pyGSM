import os
import tempfile
import weakref
import functools

from .manage_xyz import write_xyz, format_xyz, format_molden

__all__ = [
    "OutputManager",
    "XYZWriter"
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
        self.manager=output_manager
        self.format = xyz_format

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
