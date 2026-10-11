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
    sparse_rows = False
    for name in ('components_', 'exp_dirichlet_component_'):
        matrix = state[name]
        rows = []
        for row in matrix:
            values, counts = np.unique(row[::97], return_counts=True)
            fill = values[counts.argmax()]
            if np.count_nonzero(row == fill) > row.size // 2:
                coo = sparse.COO.from_numpy(row, fill_value=fill)
                coo.coords = coo.coords.astype(np.int32)
                buffer = io.BytesIO()
                sparse.save_npz(buffer, coo, compressed=False)
                rows.append(buffer.getvalue())
            else:
                rows.append(row)
        if any(isinstance(row, bytes) for row in rows):
            state[name] = rows
            sparse_rows = True
    # Byte position contexts suit float64 values; dense models prefer defaults.
    filters = [dict(id=lzma.FILTER_LZMA2, preset=9, lc=1, lp=3, pb=0)] if sparse_rows else None
    filename = Path(filename)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=filename.parent, delete=False) as f:
            temporary = Path(f.name)
            with lzma.LZMAFile(f, 'wb', filters=filters, preset=None if filters else 9) as stream:
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
        version, state = pickle.loads(lzma.decompress(f.read()))
    if version != 'lda-pydata-v1':
        raise ValueError('Unsupported LDA storage format')
    for name in ('components_', 'exp_dirichlet_component_'):
        if isinstance(state[name], list):
            state[name] = np.vstack([
                sparse.load_npz(io.BytesIO(row)).todense() if isinstance(row, bytes) else row
                for row in state[name]
            ])
    model = LatentDirichletAllocation.__new__(LatentDirichletAllocation)
    model.__setstate__(state)
    return model
