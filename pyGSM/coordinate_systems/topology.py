from __future__ import print_function
from ..utilities import manage_xyz
import numpy as np
import scipy.sparse as sparse

__all__ = [
    "EdgeGraph",
    "guess_bonds"
]

def format_xyz_block(atoms, xyz):
    return f"{len(atoms)}\n\n" + "\n".join(
        f"{a:4} {x:8.3f} {y:8.3f} {z:8.3f}"
        for a,(x,y,z) in zip(atoms, xyz)
    )
def guess_bonds(atoms, xyz, charge=0):
    import rdkit.Chem as Chem
    import rdkit.Chem.rdDetermineBonds as rdDetermineBonds

    rdmol = Chem.MolFromXYZBlock(format_xyz_block(atoms, xyz))
    if charge is None:
        charge = 0
    rdDetermineBonds.DetermineConnectivity(rdmol, charge=charge)
    return [
            [b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble()]
            for b in rdmol.GetBonds()
        ]

class EdgeGraph:
    # adapted from McUtils to fix the terrible networkx graph extension
    def __init__(self, labels, edges, edge_types=None):
        self.labels = labels
        self.label_map = {l:i for i,l in enumerate(labels)}
        self._edges = np.asanyarray(edges)
        self.graph = self.adj_mat(len(labels), self._edges)
        self._edge_types = (
            None
                if edge_types is None else
            edge_types
                if isinstance(edge_types, np.ndarray) and edge_types.ndim == 2 else
            self.type_graph(len(labels), self._edges, edge_types)
        )
        self.edge_map = self.build_edge_map(self._edges)
    def edges(self):
        return [
            (self.labels[i], self.labels[j])
            for i, j in self._edges
        ]
    def nodes(self):
        return self.labels
    def neighbors(self, a):
        return [self.labels[i] for i in self.edge_map[self.label_map[a]]]
    def edge_types(self):
        if self._edge_types is None:
            return None
        else:
            rows, cols = self._edges.T
            return self._edge_types[rows, cols]

    @classmethod
    def adj_mat(cls, num_nodes, edges):
        adj = np.zeros((num_nodes, num_nodes), dtype=int)
        if len(edges) > 0:
            rows,cols = edges.T
            adj[rows, cols] = 1
            adj[cols, rows] = 1

        return sparse.csr_matrix(adj)

    @classmethod
    def type_graph(cls, num_nodes, edges, types): # win no awards for keeping this sparse
        adj = np.zeros((num_nodes, num_nodes), dtype=float)
        if len(edges) > 0:
            rows,cols = edges.T
            adj[rows, cols] = np.array(types)
            adj[cols, rows] = np.array(types)

        return adj

    @classmethod
    def build_edge_map(cls, edge_list, num_nodes=None):
        map = {}
        for e1, e2 in edge_list:
            if e1 not in map: map[e1] = set()
            map[e1].add(e2)
            if e2 not in map: map[e2] = set()
            map[e2].add(e1)
        if num_nodes is not None:
            for i in range(num_nodes):
                if i not in map: map[i] = set()
        return map

    def get_fragment_indices(self):
        ncomp, labels = sparse.csgraph.connected_components(self.graph, directed=False, return_labels=True)
        groups = {}
        for i,l in enumerate(labels):
            if l not in groups: groups[l] = []
            groups[l].append(i)
        return list(groups.values())
    def get_fragments(self):
        return [self.take(pos) for pos in self.get_fragment_indices()]

    @classmethod
    def _remap(cls, labels, pos, rows, cols):
        if len(rows) == 0:
            edge_list = np.array([], dtype=int).reshape(-1, 2)
        else:
            new_mapping = np.zeros(len(labels), dtype=int)
            new_mapping[pos,] = np.arange(len(pos))
            new_row = new_mapping[rows,]
            new_col = new_mapping[cols,]
            edge_list = np.array([new_row, new_col]).T

        return [labels[p] for p in pos], edge_list
    @classmethod
    def _take(cls, pos, labels, adj_mat:sparse.compressed, edge_types=None) -> 'typing.Self':
        rows, cols, _ = sparse.find(adj_mat)
        utri = cols >= rows
        rows = rows[utri]
        cols = cols[utri]
        row_cont = np.isin(rows, pos)
        col_cont = np.isin(cols, pos)
        cont = np.logical_and(row_cont, col_cont)

        labels, edge_list = cls._remap(labels, pos, rows[cont], cols[cont])
        if edge_types is not None:
            edge_types = edge_types[np.ix_(rows[cont], cols[cont])]
        return cls(labels, edge_list, edge_types=edge_types)

    def take(self, pos):
        return self._take(pos, self.labels, self.graph, edge_types=self._edge_types)

    def L(self):
        """ Return a list of the sorted atom numbers in this graph. """
        return sorted(list(self.labels))

    def AStr(self):
        """ Return a string of atoms, which serves as a rudimentary 'fingerprint' : '99,100,103,151' . """
        return ','.join(['%i' % i for i in self.L()])

def AtomContact(xyz, pairs, box=None, displace=False):
    """
    Compute distances between pairs of atoms.

    Parameters
    ----------
    xyz : np.ndarray
        Nx3 array of atom positions
    pairs : list
        List of 2-tuples of atom indices
    box : np.ndarray, optional
        An array of three numbers (xyz box vectors).

    Returns
    -------
    np.ndarray
        A Npairs-length array of minimum image convention distances
    np.ndarray (optional)
        if displace=True, return a Npairsx3 array of displacement vectors
    """
    # Obtain atom selections for atom pairs
    parray = np.array(pairs)
    sel1 = parray[:, 0]
    sel2 = parray[:, 1]
    xyzpbc = xyz.copy()
    # Minimum image convention: Place all atoms in the box
    # [-xbox/2, +xbox/2); [-ybox/2, +ybox/2); [-zbox/2, +zbox/2)
    if box is not None:
        xbox = box[0]
        ybox = box[1]
        zbox = box[2]
        while any(xyzpbc[:, 0] < -0.5*xbox):
            xyzpbc[:, 0] += (xyzpbc[:, 0] < -0.5*xbox)*xbox
        while any(xyzpbc[:, 1] < -0.5*ybox):
            xyzpbc[:, 1] += (xyzpbc[:, 1] < -0.5*ybox)*ybox
        while any(xyzpbc[:, 2] < -0.5*zbox):
            xyzpbc[:, 2] += (xyzpbc[:, 2] < -0.5*zbox)*zbox
        while any(xyzpbc[:, 0] >= 0.5*xbox):
            xyzpbc[:, 0] -= (xyzpbc[:, 0] >= 0.5*xbox)*xbox
        while any(xyzpbc[:, 1] >= 0.5*ybox):
            xyzpbc[:, 1] -= (xyzpbc[:, 1] >= 0.5*ybox)*ybox
        while any(xyzpbc[:, 2] >= 0.5*zbox):
            xyzpbc[:, 2] -= (xyzpbc[:, 2] >= 0.5*zbox)*zbox
    # Obtain atom selections for the pairs to be computed
    # These are typically longer than N but shorter than N^2.
    xyzsel1 = xyzpbc[sel1]
    xyzsel2 = xyzpbc[sel2]

    # print("sel1, sel2")
    # print(sel1)
    # print(sel2)
    # Calculate xyz displacement
    dxyz = xyzsel2-xyzsel1
    # Apply minimum image convention to displacements
    if box is not None:
        dxyz[:, 0] += (dxyz[:, 0] < -0.5*xbox)*xbox
        dxyz[:, 1] += (dxyz[:, 1] < -0.5*ybox)*ybox
        dxyz[:, 2] += (dxyz[:, 2] < -0.5*zbox)*zbox
        dxyz[:, 0] -= (dxyz[:, 0] >= 0.5*xbox)*xbox
        dxyz[:, 1] -= (dxyz[:, 1] >= 0.5*ybox)*ybox
        dxyz[:, 2] -= (dxyz[:, 2] >= 0.5*zbox)*zbox
    dr2 = np.sum(dxyz**2, axis=1)
    dr = np.sqrt(dr2)
    if displace:
        return dr, dxyz
    else:
        return dr


if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    #filepath='../../data/butadiene_ethene.xyz'
    #filepath='crystal.xyz'
    filepath1 = 'multi1.xyz'
    filepath2 = 'multi2.xyz'

    geom1 = manage_xyz.read_xyz(filepath1)
    geom2 = manage_xyz.read_xyz(filepath2)
    atom_symbols = manage_xyz.get_atoms(geom1)
    xyz1 = manage_xyz.xyz_to_np(geom1)
    xyz2 = manage_xyz.xyz_to_np(geom2)

    ELEMENT_TABLE = elements.ElementData()
    atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in atom_symbols]
    #print(atoms)

    #hybrid_indices = list(range(0,10)) + list(range(21,26))
    hybrid_indices = list(range(16, 26))

    G1 = Topology.build_topology(xyz1, atoms, hybrid_indices=hybrid_indices)
    G2 = Topology.build_topology(xyz2, atoms, hybrid_indices=hybrid_indices)

    for bond in G2.edges():
        if bond in G1.edges:
            pass
        elif (bond[1], bond[0]) in G1.edges():
            pass
        else:
            print(" Adding bond {} to top1".format(bond))
            if bond[0] > bond[1]:
                G1.add_edge(bond[0], bond[1])
            else:
                G1.add_edge(bond[1], bond[0])

    #print(" G")
    #print(G.L())

    #fragments = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    #for g in fragments: g.__class__ = MyG

    #print(" fragments")
    #for frag in fragments:
    #    print(frag.L())
    ##print(len(mytop.fragments))
    ##print(mytop.fragments)

    ## need the primitive start and stop indices
    #prim_idx_start_stop=[]
    #new=True
    #for frag in fragments:
    #    nodes=frag.L()
    #    prim_idx_start_stop.append((nodes[0],nodes[-1]))
    #print("prim start stop")
    #print(prim_idx_start_stop)

    #prim_idx =[]
    #for info in prim_idx_start_stop:
    #    prim_idx += list(range(info[0],info[1]+1))
    #print('prim_idx')
    #print(prim_idx)

    #new_hybrid_indices=list(range(len(atoms)))
    #for elem in prim_idx:
    #    new_hybrid_indices.remove(elem)
    #print('hybr')
    #print(new_hybrid_indices)

    #hybrid_idx_start_stop=[]
    ## get the hybrid start and stop indices
    #new=True
    #for i in range(len(atoms)+1):
    #    if i in new_hybrid_indices:
    #        print(i)
    #        if new==True:
    #            start=i
    #            new=False
    #    else:
    #        if new==False:
    #            end=i-1
    #            new=True
    #            hybrid_idx_start_stop.append((start,end))
    #print(hybrid_idx_start_stop)
