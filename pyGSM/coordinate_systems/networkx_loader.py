from ..utilities import nifty
try:
    import networkx as nx
except ImportError:
    nx = None
    nifty.logger.warning("NetworkX cannot be imported (topology tools won't work).  Most functionality should still work though.")