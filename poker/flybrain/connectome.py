"""Loader for the MaleCNS v1.0 connectome.

Data source
-----------
MaleCNS v1.0 (adult male Drosophila central nervous system), HHMI Janelia Research
Campus and Google Research, released 2026-09-03. Licensed CC-BY, which permits
commercial use with attribution. Bulk files live in a public Google Cloud Storage
bucket; this module pulls only the three we need:

    body-annotations-...feather       14 MB   cell class / type per neuron
    body-neurotransmitters-...feather 42 MB   predicted transmitter per neuron
    connectome-weights-...feather    1.0 GB   body_pre, body_post, weight

The 6.8 GB synaptic-partner and 12.7 GB synapse-point files are not needed: the
segment-to-segment weight table already is the adjacency matrix.

Memory note
-----------
The weight table has 151,856,684 rows. Materialising boolean masks over all of it at
once exhausts a normal container, so every pass here is batched and uses searchsorted
membership rather than np.isin.
"""

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
import logging
import os
import urllib.request

import numpy as np

from poker.flybrain.config import MB_CLASSES, NT_SIGN, Scope

log = logging.getLogger(__name__)

BUCKET = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome'
FILES = {
    'annotations': 'body-annotations-male-cns-v1.0-minconf-0.5.feather',
    'neurotransmitters': 'body-neurotransmitters-male-cns-v1.0.feather',
    'weights': 'connectome-weights-male-cns-v1.0-minconf-0.5.feather',
}
ATTRIBUTION = ('MaleCNS v1.0 connectome, HHMI Janelia Research Campus & Google Research, '
               'CC-BY. https://male-cns.janelia.org/')

_BATCH_ROWS = 4_000_000


class Connectome:
    """Neurons, their cell classes, transmitter signs, and a sparse weight matrix.

    Attributes:
        body_ids: int64 array of neuron ids, sorted. Index into this is the neuron index.
        classes: object array of MaleCNS cell class per neuron ('Kenyon_Cell', 'MBON', ...).
        types: object array of cell type per neuron ('KCg-m', 'PPL101', 'ORN_DA1', ...).
        signs: float32 array, +1 excitatory / -1 inhibitory / 0 modulatory.
        weights: scipy.sparse CSR matrix, shape (n, n), signed. weights[i, j] is the
            drive neuron i delivers to neuron j.
        synthetic: True when this is the structurally-matched stand-in, not real data.
    """

    def __init__(self, body_ids, classes, types, signs, weights, is_synthetic=False):
        self.body_ids = body_ids
        self.classes = classes
        self.types = types
        self.signs = signs
        self.weights = weights
        self.synthetic = is_synthetic

    @property
    def n_neurons(self):
        """Number of neurons in the loaded scope."""
        return len(self.body_ids)

    @property
    def n_edges(self):
        """Number of non-zero connections."""
        return int(self.weights.nnz)

    def index_of_class(self, *class_names):
        """Return neuron indices whose cell class is any of class_names."""
        mask = np.isin(self.classes, np.asarray(class_names, dtype=object))
        return np.flatnonzero(mask)

    def index_of_type_prefix(self, *prefixes):
        """Return neuron indices whose cell type starts with any of prefixes.

        Used to pull out DAN clusters: PAM* are the appetitive ones, PPL1* the
        aversive ones (including PPL101, the pair DOOMFLY wired to damage).
        """
        types = np.asarray([t if isinstance(t, str) else '' for t in self.types])
        mask = np.zeros(len(types), dtype=bool)
        for prefix in prefixes:
            mask |= np.char.startswith(types, prefix)
        return np.flatnonzero(mask)

    def describe(self):
        """One-line summary for logs."""
        kind = 'synthetic' if self.synthetic else 'MaleCNS v1.0'
        return f"{kind}: {self.n_neurons:,} neurons, {self.n_edges:,} edges"


def default_cache_dir():
    """Where downloaded and preprocessed connectome data lives (gitignored)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), 'data', 'flybrain')


def _download(name, cache_dir, progress=True):
    """Fetch one bulk file if it is not already cached. Returns the local path."""
    target = os.path.join(cache_dir, FILES[name])
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target
    os.makedirs(cache_dir, exist_ok=True)
    url = f"{BUCKET}/{FILES[name]}"
    tmp = target + '.part'
    log.info("Downloading %s -> %s", url, target)

    def _hook(blocks, block_size, total):
        if not progress or total <= 0:
            return
        pct = min(100.0, 100.0 * blocks * block_size / total)
        if int(pct) % 10 == 0:
            log.info("  %s: %.0f%%", name, pct)

    urllib.request.urlretrieve(url, tmp, reporthook=_hook)  # nosec B310 - fixed https host
    os.replace(tmp, target)
    return target


def _read_feather(path, columns=None):
    """Read a feather file, preferring pyarrow and falling back to pandas."""
    try:
        from pyarrow import feather  # pylint: disable=import-outside-toplevel
        return feather.read_table(path, columns=columns)
    except ImportError as exc:
        raise ImportError(
            "Reading the MaleCNS feather files needs pyarrow. Install it with "
            "`pip install pyarrow`, or run with use_synthetic=True."
        ) from exc


def _load_annotations(cache_dir):
    """Return (body_ids, classes, types) for Traced neurons, sorted by body id.

    'Traced' is the status for a fully reconstructed neuron: 165,122 of them. The
    other statuses (Orphan, Glia, Unimportant) are fragments or non-neurons.
    """
    path = _download('annotations', cache_dir)
    table = _read_feather(path, columns=['bodyId', 'class', 'type', 'status'])
    body = table.column('bodyId').to_numpy()
    status = np.asarray(table.column('status').to_pylist(), dtype=object)
    cls = np.asarray(table.column('class').to_pylist(), dtype=object)
    typ = np.asarray(table.column('type').to_pylist(), dtype=object)

    keep = status == 'Traced'
    body, cls, typ = body[keep], cls[keep], typ[keep]
    order = np.argsort(body, kind='stable')
    return body[order].astype(np.int64), cls[order], typ[order]


def _load_signs(cache_dir, body_ids):
    """Map each neuron to a postsynaptic sign from its predicted transmitter.

    The neurotransmitter table has one row per (body, cell_type) pair, so a body can
    appear many times. We take the most confident non-'unclear' consensus call per
    body and fall back to the excitatory majority when every call is unclear.
    """
    path = _download('neurotransmitters', cache_dir)
    table = _read_feather(path, columns=['body', 'consensus_nt', 'predicted_nt_confidence'])
    bodies = table.column('body').to_numpy()
    nts = np.asarray(table.column('consensus_nt').to_pylist(), dtype=object)
    conf = np.nan_to_num(table.column('predicted_nt_confidence').to_numpy(zero_copy_only=False))

    # Strongest evidence last, so the final write per body wins.
    informative = np.asarray([nt not in (None, 'unclear') for nt in nts])
    order = np.lexsort((conf, informative))

    idx = np.searchsorted(body_ids, bodies[order])
    np.clip(idx, 0, len(body_ids) - 1, out=idx)
    valid = body_ids[idx] == bodies[order]

    resolved = np.full(len(body_ids), 'unclear', dtype=object)
    resolved[idx[valid]] = nts[order][valid]

    signs = np.asarray(
        [NT_SIGN.get(nt if isinstance(nt, str) else 'unclear', 1.0) for nt in resolved],
        dtype=np.float32,
    )
    counts = {nt: int((resolved == nt).sum()) for nt in set(resolved)}
    log.info("Transmitter calls: %s", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
    return signs


def _load_edges(cache_dir, body_ids, weight_threshold):
    """Scan the weight table in batches and return (rows, cols, vals) within body_ids.

    Returns indices into body_ids, not raw body ids.
    """
    path = _download('weights', cache_dir)
    table = _read_feather(path)

    def membership(values):
        idx = np.searchsorted(body_ids, values)
        np.clip(idx, 0, len(body_ids) - 1, out=idx)
        return idx, body_ids[idx] == values

    rows, cols, vals = [], [], []
    scanned = 0
    for batch in table.to_batches(max_chunksize=_BATCH_ROWS):
        pre = batch.column('body_pre').to_numpy()
        post = batch.column('body_post').to_numpy()
        wt = batch.column('weight').to_numpy()

        heavy = wt >= weight_threshold
        if not heavy.any():
            scanned += len(wt)
            continue
        pre, post, wt = pre[heavy], post[heavy], wt[heavy]

        pre_idx, pre_ok = membership(pre)
        post_idx, post_ok = membership(post)
        keep = pre_ok & post_ok
        if keep.any():
            rows.append(pre_idx[keep].astype(np.int32))
            cols.append(post_idx[keep].astype(np.int32))
            vals.append(wt[keep].astype(np.float32))
        scanned += len(heavy)

    log.info("Scanned %s edges, kept %s at weight>=%d",
             f"{scanned:,}", f"{sum(len(v) for v in vals):,}", weight_threshold)
    if not rows:
        return (np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def _build_matrix(rows, cols, vals, signs, n, max_abs_weight):
    """Assemble the signed sparse weight matrix."""
    from scipy import sparse  # pylint: disable=import-outside-toplevel

    vals = np.minimum(vals, max_abs_weight)
    # A synapse's sign is set by the presynaptic neuron's transmitter.
    signed = vals * signs[rows]
    matrix = sparse.coo_matrix((signed, (rows, cols)), shape=(n, n), dtype=np.float32)
    return matrix.tocsr()


def _cache_path(cache_dir, scope, weight_threshold):
    return os.path.join(cache_dir, f"connectome-{scope.value}-w{weight_threshold}.npz")


def load(scope=Scope.mushroom_body, weight_threshold=3, cache_dir=None,
         use_synthetic=False, rebuild=False, seed=0):
    """Load a Connectome, downloading and preprocessing on first use.

    Args:
        scope: Scope.mushroom_body (8,246 neurons) or Scope.whole_brain (165,122).
        weight_threshold: drop connections below this synapse count. 3 is the usual
            floor for treating an EM-reconstructed connection as real.
        cache_dir: where to keep downloads and the preprocessed npz.
        use_synthetic: skip the real data and build a structurally-matched stand-in.
        rebuild: ignore any preprocessed cache.
        seed: seed for the synthetic fallback.

    Returns:
        Connectome
    """
    cache_dir = cache_dir or default_cache_dir()
    if use_synthetic:
        return synthetic(scope=scope, seed=seed)

    npz = _cache_path(cache_dir, scope, weight_threshold)
    if os.path.exists(npz) and not rebuild:
        log.info("Loading preprocessed connectome from %s", npz)
        return _load_npz(npz)

    try:
        body_ids, classes, types = _load_annotations(cache_dir)
    except (OSError, ImportError) as exc:
        log.warning("Could not load MaleCNS annotations (%s). Falling back to a "
                    "synthetic connectome - results are NOT based on real wiring.", exc)
        return synthetic(scope=scope, seed=seed)

    if scope is Scope.mushroom_body:
        keep = np.isin(classes, np.asarray(MB_CLASSES, dtype=object))
        body_ids, classes, types = body_ids[keep], classes[keep], types[keep]
    log.info("Scope %s: %s neurons", scope.value, f"{len(body_ids):,}")

    signs = _load_signs(cache_dir, body_ids)
    rows, cols, vals = _load_edges(cache_dir, body_ids, weight_threshold)
    matrix = _build_matrix(rows, cols, vals, signs, len(body_ids), max_abs_weight=400.0)

    conn = Connectome(body_ids, classes, types, signs, matrix)
    _save_npz(npz, conn)
    log.info("Built %s", conn.describe())
    return conn


def _save_npz(path, conn):
    """Cache a built connectome so the 27s full scan happens only once."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    matrix = conn.weights.tocoo()
    np.savez_compressed(
        path,
        body_ids=conn.body_ids,
        classes=np.asarray([c if isinstance(c, str) else '' for c in conn.classes]),
        types=np.asarray([t if isinstance(t, str) else '' for t in conn.types]),
        signs=conn.signs,
        row=matrix.row.astype(np.int32),
        col=matrix.col.astype(np.int32),
        val=matrix.data.astype(np.float32),
        synthetic=np.asarray([conn.synthetic]),
    )
    log.info("Cached connectome to %s", path)


def _load_npz(path):
    """Restore a cached connectome."""
    from scipy import sparse  # pylint: disable=import-outside-toplevel

    data = np.load(path, allow_pickle=False)
    n = len(data['body_ids'])
    matrix = sparse.coo_matrix(
        (data['val'], (data['row'], data['col'])), shape=(n, n), dtype=np.float32
    ).tocsr()
    return Connectome(
        data['body_ids'],
        data['classes'].astype(object),
        data['types'].astype(object),
        data['signs'],
        matrix,
        is_synthetic=bool(data['synthetic'][0]),
    )


def synthetic(scope=Scope.mushroom_body, seed=0):
    """Build a stand-in connectome with the real class composition and edge density.

    This exists so the whole pipeline - encoding, LIF, decoding, plasticity, tests -
    can run before the 1 GB download finishes, and so CI does not need the dataset.
    It is NOT the fly's wiring: any result produced on it says nothing about the
    connectome. Connectome.synthetic is True so callers can refuse to report it.

    Population sizes and edge counts match what MaleCNS v1.0 actually yields at
    weight>=3 (measured, not guessed).
    """
    from scipy import sparse  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(seed)
    if scope is Scope.mushroom_body:
        composition = [
            ('olfactory', 'ORN_synth', 2639),
            ('ALPN', 'PN_synth', 686),
            ('ALLN', 'LN_synth', 420),
            ('Kenyon_Cell', 'KC_synth', 4064),
            ('MBON', 'MBON_synth', 97),
            ('DAN', 'PAM_synth', 240),
            ('DAN', 'PPL1_synth', 100),
        ]
        n_edges = 451_855
    else:
        composition = [('cb_intrinsic', 'synth', 165_122)]
        n_edges = 10_511_038

    classes, types = [], []
    for cls, typ, count in composition:
        classes.extend([cls] * count)
        types.extend([f"{typ}{i:05d}" for i in range(count)])
    n = len(classes)

    body_ids = np.arange(1, n + 1, dtype=np.int64)
    # Roughly the real transmitter mix: mostly cholinergic, some GABA/glutamate.
    signs = rng.choice([1.0, -1.0], size=n, p=[0.75, 0.25]).astype(np.float32)

    rows = rng.integers(0, n, size=n_edges, dtype=np.int32)
    cols = rng.integers(0, n, size=n_edges, dtype=np.int32)
    vals = (3.0 + rng.exponential(8.0, size=n_edges)).astype(np.float32)
    self_loops = rows == cols
    rows, cols, vals = rows[~self_loops], cols[~self_loops], vals[~self_loops]

    matrix = sparse.coo_matrix(
        (vals * signs[rows], (rows, cols)), shape=(n, n), dtype=np.float32
    ).tocsr()

    log.warning("Using a SYNTHETIC connectome (%s neurons). Real wiring is not loaded; "
                "no claim about the fly's circuit can be made from these runs.", f"{n:,}")
    return Connectome(body_ids, np.asarray(classes, dtype=object),
                      np.asarray(types, dtype=object), signs, matrix, is_synthetic=True)
