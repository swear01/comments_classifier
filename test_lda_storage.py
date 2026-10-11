import pickle
import lzma
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation

from lda_storage import load_model, save_model


class StorageTests(unittest.TestCase):
    def test_state_bits_predictions_and_source_unchanged(self):
        x = csr_matrix([[1., 0., 3., 0.], [0., 2., 0., 4.]])
        model = LatentDirichletAllocation(n_components=2, random_state=0).fit(x)
        for dense in (False, True):
            model.components_ = np.full((2, 32), .01)
            model.exp_dirichlet_component_ = np.full((2, 32), .02)
            if dense:
                order = np.random.RandomState(12).permutation(64).reshape(2, 32)
                model.components_ += order * np.spacing(.01)
                model.exp_dirichlet_component_ += order * np.spacing(.02)
            else:
                model.components_[:, -1] = np.nextafter(.01, np.inf)
                model.exp_dirichlet_component_[:, -1] = np.nextafter(.02, np.inf)
            before = pickle.dumps(model, protocol=4)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'model.pickle'
                save_model(model, path)
                restored = load_model(path)
                complete = path.read_bytes()
                path.write_bytes(complete[:-4])
                with self.assertRaises(lzma.LZMAError):
                    load_model(path)
                path.write_bytes(complete)
                self.assertEqual(before, pickle.dumps(model, protocol=4))
                for name in ('components_', 'exp_dirichlet_component_'):
                    a, b = getattr(model, name), getattr(restored, name)
                    self.assertEqual(a.dtype, b.dtype)
                    self.assertEqual(a.tobytes(), b.tobytes())
                for name in vars(model).keys() - {'components_', 'exp_dirichlet_component_', 'random_state_'}:
                    np.testing.assert_equal(getattr(model, name), getattr(restored, name))
                np.testing.assert_equal(model.random_state_.get_state(), restored.random_state_.get_state())
                # Match the modified matrix width when exercising both models.
                sample = csr_matrix(([1., 2.], ([0, 0], [0, 1])), shape=(1, 32))
                model.n_features_in_ = restored.n_features_in_ = 32
                np.testing.assert_array_equal(model.transform(sample), restored.transform(sample))
                path.write_bytes(before)
                self.assertIsInstance(load_model(path), LatentDirichletAllocation)

    def test_failed_save_preserves_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pickle'
            path.write_bytes(b'original')
            with self.assertRaises(KeyError):
                save_model(LatentDirichletAllocation(), path)
            self.assertEqual(path.read_bytes(), b'original')
            model = LatentDirichletAllocation(n_components=2, random_state=0).fit([[1., 2.], [2., 1.]])
            model.unpicklable = lambda: None
            with self.assertRaises((AttributeError, pickle.PicklingError)):
                save_model(model, path)
            self.assertEqual(path.read_bytes(), b'original')
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
