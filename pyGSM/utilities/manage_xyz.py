import numpy as np

from . import Devutils as dev
from . import units
import re
#import openbabel as ob

# => XYZ File Utility <= #


def read_xyz(
        filename,
        scale=1.):
    """ Read xyz file

    Params:
        filename (str) - name of xyz file to read

    Returns:
        geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)

    """

    lines = dev.read_file(filename).splitlines()
    lines = lines[2:]
    geom = []
    for line in lines:
        mobj = re.match(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', line)
        geom.append((
            mobj.group(1),
            scale*float(mobj.group(2)),
            scale*float(mobj.group(3)),
            scale*float(mobj.group(4)),
        ))
    return geom


def read_xyzs(
    filename,
    scale=1.
):
    """ Read xyz file

    Params:
        filename (str) - name of xyz file to read

    Returns:
        geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)

    """

    lines = dev.read_file(filename).splitlines()
    natoms = int(lines[0])
    total_lines = len(lines)
    num_geoms = total_lines/(natoms+2)

    geoms = []
    sa = 2
    for i in range(int(num_geoms)):
        ea = sa+natoms
        geom = []
        for line in lines[sa:ea]:
            mobj = re.match(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', line)
            geom.append((
                mobj.group(1),
                scale*float(mobj.group(2)),
                scale*float(mobj.group(3)),
                scale*float(mobj.group(4)),
            ))
        sa = ea+2
        geoms.append(geom)
    return geoms


def read_molden_geoms(
    filename,
    scale=1.
):

    lines = dev.read_file(filename).splitlines()
    natoms = int(lines[2])
    nlines = len(lines)

    # this is for three blocks after GEOCON
    num_geoms = (nlines-6) / (natoms+5)
    num_geoms = int(num_geoms)
    print(num_geoms)
    geoms = []

    sa = 4
    for i in range(int(num_geoms)):
        ea = sa+natoms
        geom = []
        for line in lines[sa:ea]:
            mobj = re.match(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', line)
            geom.append((
                mobj.group(1),
                scale*float(mobj.group(2)),
                scale*float(mobj.group(3)),
                scale*float(mobj.group(4)),
            ))
        sa = ea+2
        geoms.append(geom)
    return geoms


def read_molden_Energy(
        filename,
):
    with open(filename) as f:
        nlines = sum(1 for _ in f)
    # print "number of lines is ", nlines
    with open(filename) as f:
        natoms = int(f.readlines()[2])

    # print "number of atoms is ",natoms
    nstructs = (nlines-6) / (natoms+5)  # this is for three blocks after GEOCON
    nstructs = int(nstructs)

    # print "number of structures in restart file is %i" % nstructs
    coords = []
    E = [0.]*nstructs
    grmss = []
    atomic_symbols = []
    dE = []
    with open(filename) as f:
        f.readline()
        f.readline()  # header lines
        # get coords
        for struct in range(nstructs):
            tmpcoords = np.zeros((natoms, 3))
            f.readline()  # natoms
            f.readline()  # space
            for a in range(natoms):
                line = f.readline()
                tmp = line.split()
                tmpcoords[a, :] = [float(i) for i in tmp[1:]]
                if struct == 0:
                    atomic_symbols.append(tmp[0])
        coords.append(tmpcoords)
        # Get energies
        f.readline()  # line
        f.readline()  # energy
        for struct in range(nstructs):
            E[struct] = float(f.readline())

    return E


def write_molden_geoms(
        filename,
        geoms,
        energies,
        gradrms,
        dEs,
):
    with open(filename, 'w') as f:
        f.write("[Molden Format]\n[Geometries] (XYZ)\n")
        for geom in geoms:
            f.write('%d\n\n' % len(geom))
            for atom in geom:
                f.write('%-2s %14.6f %14.6f %14.6f\n' % (
                    atom[0],
                    atom[1],
                    atom[2],
                    atom[3],
                ))
        f.write("[GEOCONV]\n")
        f.write('energy\n')
        V0 = energies[0]
        for energy in energies:
            f.write('{}\n'.format(energy-V0))
        f.write("max-force\n")
        for grad in gradrms:
            f.write('{}\n'.format(float(grad)))
        # rint(" WARNING: Printing dE as max-step in molden output ")
        f.write("max-step\n")
        for dE in dEs:
            f.write('{}\n'.format(float(dE)))


def get_atoms(
        geom,
):

    atoms = []
    for atom in geom:
        atoms.append(atom[0])
    return atoms

def format_xyz(atoms, coords,
                # '%-2s %14.6f %14.6f %14.6f\n'
                line_fmt="{atom:3>} {x:14.6f} {y:14.6f} {z:14.6f}",
                scale=1.0,
                comment:str|list[str]=0,
                include_header=True,
                validate=True):

    coords = scale * np.asanyarray(coords)
    if validate:
        if not isinstance(atoms[0], str):
            atoms = [a.symbol for a in atoms]
        if coords.ndim == 1:
            raise ValueError("coords must be an nx3 array (got {coords.shape})")
        if coords.ndim > 3:
            raise ValueError(f"coords must be a stack of nx3 arrays (got {coords.shape})")
        if coords.shape[-2] != len(atoms):
            raise ValueError(f"coords of shape {coords.shape} don't work with {len(atoms)} atoms")
    if coords.ndim == 3:
        if hasattr(comment, '__getitem__'):
            if len(comment) != len(coords):
                raise ValueError(f"different number of comments {len(comment)} than coords {len(coords)}")
        else:
            comment = [str(comment) + f" - {i}" for i in range(len(coords))]
        blocks = [
            _format_xyz(atoms,
                        c,
                        comment=com,
                        include_header=include_header,
                        validate=False)
            for i,(c,com) in enumerate(zip(coords, comment))
        ]
        return "\n\n".join(blocks)
    else:
        if include_header:
            lines = [
                str(len(atoms)),
                str(comment)
            ]
        else:
            lines = []
        lines = lines + [
            line_fmt.format(atom=a, x=x, y=y, z=z)
            for a, (x, y, z) in zip(atoms, coords)
        ]
        return "\n".join(lines)

def format_molden(atoms, coords,
                   energies,
                   gradrms,
                   dEs,
                   line_fmt="{atom:3>} {x:14.6f} {y:14.6f} {z:14.6f}",
                   scale=1.0,
                   comment=None,
                   format_header="[Molden Format]\n[Geometries] (XYZ)",
                   include_header=True,
                   validate=True):
    energies = np.asanyarray(energies)
    if comment is None:
        comment = [f"Energy: {e}" for e in energies]
    base = format_xyz(
        atoms,
        coords,
        line_fmt,
        scale=scale,
        comment=comment,
        include_header=include_header,
        validate=validate
    )

    footer = [
        "[GEOCONV]",
        "energy"
    ] + [
        "{:16.12f}".format(e)
        for e in energies - energies[0]
    ] + ["max-force"] + [
        "{:16.12f}".format(g)
        for g in gradrms
    ] + ["max-step"] + [
        "{:16.12f}".format(dE)
        for dE in dEs
    ]

    return "\n".join([
        format_header,
        base
        ] + footer)

def write_xyz(filename_or_object, atoms, geom, *etc, formatter=None, mode='w+', **format_opts):
    if formatter is None:
        formatter = format_xyz
    return dev.write_file(
        filename_or_object,
        formatter(atoms, geom, *etc, **format_opts),
        mode=mode
    )


# def write_xyz(
#     filename,
#     geom,
#     comment=0,
#     scale=1.0  # (1.0/units.ANGSTROM_TO_AU),
# ):
#     """ Writes xyz file with single frame
#
#     Params:
#         filename (str) - name of xyz file to write
#         geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)
#
#     """
#     with open(filename, 'w') as fh:
#         fh.write('%d\n' % len(geom))
#         fh.write('{}\n'.format(comment))
#         for atom in geom:
#             fh.write('%-2s %14.6f %14.6f %14.6f\n' % (
#                 atom[0],
#                 scale*atom[1],
#                 scale*atom[2],
#                 scale*atom[3],
#             ))
#
#
# def write_xyzs(
#     filename,
#     geoms,
#     scale=1.,
# ):
#     """ Writes xyz trajectory file with multiple frames
#
#     Params:
#         filename (str) - name of xyz file to write
#         geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)
#
#     Returns:
#
#     """
#
#     with open(filename, 'w') as fh:
#         for geom in geoms:
#             fh.write('%d\n\n' % len(geom))
#             for atom in geom:
#                 fh.write('%-2s %14.6f %14.6f %14.6f\n' % (
#                     atom[0],
#                     scale*atom[1],
#                     scale*atom[2],
#                     scale*atom[3],
#                 ))
#
#
# def write_std_multixyz(
#         filename,
#         geoms,
#         energies,
#         gradrms,
#         dEs,
# ):
#     with open(filename, 'w') as f:
#         for E, geom in zip(energies, geoms):
#             f.write('%d\n' % len(geom))
#             f.write('%.6f\n' % (E*units.KJ_MOL_TO_AU))
#             for atom in geom:
#                 f.write('%-2s %14.6f %14.6f %14.6f\n' % (
#                     atom[0],
#                     atom[1],
#                     atom[2],
#                     atom[3],
#                 ))
#
#
# def write_amber_xyz(
#         filename,
#         geom,
# ):
#
#     count = 0
#     with open(filename, 'w') as fh:
#         fh.write("default name\n")
#         fh.write('  %d\n' % len(geom))
#         for line in geom:
#             for elem in line[1:]:
#                 fh.write(" {:11.7f}".format(float(elem)))
#                 count += 1
#             if count % 6 == 0:
#                 fh.write("\n")
#
#
# def write_xyzs_w_comments(
#     filename,
#     geoms,
#     comments,
#     scale=1.0  # (1.0/units.ANGSTROM_TO_AU),
# ):
#     """ Writes xyz trajectory file with multiple frames
#
#     Params:
#         filename (str) - name of xyz file to write
#         geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)
#
#     Returns:
#
#     """
#
#     with open(filename, 'w') as fh:
#         for geom, comment in zip(geoms, comments):
#             fh.write('%d\n' % len(geom))
#             fh.write('%s\n' % comment)
#             for atom in geom:
#                 fh.write('%-2s %14.6f %14.6f %14.6f\n' % (
#                     atom[0],
#                     scale*atom[1],
#                     scale*atom[2],
#                     scale*atom[3],
#                 ))
#
#
def xyz_to_np(
    geom,
):
    """ Convert from xyz file format xyz array for editing geometry

    Params:
        geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)

    Returns:
        xyz ((natoms,3) np.ndarray) - system geometry (x,y,z)

    """

    xyz2 = np.zeros((len(geom), 3))
    for A, atom in enumerate(geom):
        xyz2[A, 0] = atom[1:]
    return xyz2
#
#
def np_to_xyz(
    geom,
    xyz2,
):
    """ Convert from xyz array to xyz file format in order to write xyz

    Params:
        geom ((natoms,4) np.ndarray) - system reference geometry
            (atom symbol, x,y,z) from xyz file
        xyz2 ((natoms,3) np.ndarray) - system geometry (x,y,z)

    Returns:
        geom2 ((natoms,4) np.ndarray) - new system geometry
            (atom symbol, x,y,z)

    """

    return [
        (a[0], x, y, z)
        for a,(x,y,z) in zip(geom, xyz2)
    ]
#
#
# def combine_atom_xyz(
#     atoms,
#     xyz,
# ):
#     """ Combines atom list with xyz array
#      Params:
#         atom list
#         geom ((natoms,3) np.ndarray) - system geometry (atom symbol, x,y,z)
#
#     Returns:
#         geom2 ((natoms,4) np.ndarray) - new system geometry
#             (atom symbol, x,y,z)
#
#     """
#     geom2 = []
#     for A, atom in enumerate(atoms):
#         geom2.append((
#             atom,
#             xyz[A, 0],
#             xyz[A, 1],
#             xyz[A, 2],
#         ))
#     return geom2
#
#
# def write_fms90(
#     filename,
#     geomx,
#     geomp=None,
# ):
#     """ Write fms90 geometry file with position and velocities
#
#     Params:
#         filename (str) - name of fms90 geometry file to write
#         geomx ((natoms,4) np.ndarray) - system positions (atom symbol, x,y,z)
#         geomp ((natoms,4) np.ndarray) - system momenta
#             (atom symbol, px, py, pz)
#
#     """
#     with open(filename, 'w') as fh:
#         fh.write('UNITS=BOHR\n')
#         fh.write('%d\n' % len(geomx))
#         for atom in geomx:
#             fh.write('%-2s %14.6f %14.6f %14.6f\n' % (
#                 atom[0],
#                 atom[1],
#                 atom[2],
#                 atom[3],
#             ))
#         if geomp:
#             fh.write('# momenta\n')
#             for atom in geomp:
#                 fh.write('  %14.6f %14.6f %14.6f\n' % (
#                     atom[1],
#                     atom[2],
#                     atom[3],
#                 ))
#
#
# XYZ_WRITERS = {
#     'molden': write_molden_geoms,
#     'multixyz': write_std_multixyz,
# }
