import io
import lzma
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import sparse
from sklearn.decomposition import LatentDirichletAllocation


def _write_state(state, filename):
    state = state.copy()
    for name in ('components_', 'exp_dirichlet_component_'):
        matrix = state[name]
        rows = []
        for row in matrix:
            values, counts = np.unique(row, return_counts=True)
            fill = values[counts.argmax()]
            coo = sparse.COO.from_numpy(row, fill_value=fill)
            # Sorting values helps LZMA; coordinates preserve their positions.
            order = np.argsort(coo.data, kind='stable')
            buffer = io.BytesIO()
            np.savez(buffer, data=coo.data[order], shape=coo.shape,
                     fill_value=coo.fill_value, coords=coo.coords[:, order].astype(np.int32))
            rows.append(buffer.getvalue())
        state[name] = rows
    filters = [dict(id=lzma.FILTER_LZMA2, preset=9, lc=1, lp=3, pb=0)]
    filename = Path(filename)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=filename.parent, delete=False) as f:
            temporary = Path(f.name)
            with lzma.LZMAFile(f, 'wb', filters=filters) as stream:
                pickle.dump(('lda-pydata-v1', state), stream, protocol=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, filename)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_model(model, filename):
    _write_state(model.__getstate__(), filename)


def load_model(filename):
    with open(filename, 'rb') as f:
        compressed = f.read(6) == b'\xfd7zXZ\x00'
        f.seek(0)
        if not compressed:
            return pickle.load(f)
        with lzma.LZMAFile(f, 'rb') as stream:
            version, state = pickle.load(stream)
            # Read through the XZ footer to verify its checksum and completeness.
            while stream.read(1024 * 1024):
                pass
    if version != 'lda-pydata-v1':
        raise ValueError('Unsupported LDA storage format')
    for name in ('components_', 'exp_dirichlet_component_'):
        if isinstance(state[name], list):
            rows = []
            for row in state[name]:
                if isinstance(row, bytes):
                    with np.load(io.BytesIO(row), allow_pickle=False) as data:
                        row = sparse.COO(coords=data['coords'], data=data['data'],
                                         shape=tuple(data['shape']), fill_value=data['fill_value'][()],
                                         sorted=False, has_duplicates=False).todense()
                rows.append(row)
            state[name] = np.vstack(rows)
    model = LatentDirichletAllocation.__new__(LatentDirichletAllocation)
    model.__setstate__(state)
    return model
