# LDA model storage

`lda.py` saves models with `lda_storage.save_model`; `lda_abstract.py` reads
them with `lda_storage.load_model`. Filenames stay unchanged. Rows use
PyData/Sparse COO with an exact fill value. Stored values are sorted together
with their coordinates to improve compression without changing their bits
or positions. Uncompressed sparse NPZ payloads and the complete model state
are compressed together with LZMA2 (preset 9, lc=1, lp=3, pb=0).
Both float64 matrices, all parameters and the random
state are retained. Loading restores NumPy arrays before inference.

Install `sparse` in the environment that runs the project. The storage reader
was tested with Sparse 0.12.0 and 0.19.2. Keep the scikit-learn version used to
train each model: the archived models were saved with **0.22.1**. Compression
does not upgrade them or remove that requirement. Load only trusted files;
these files still contain pickle data. Reading ordinary original pickles is
also supported.

`lda_abstract.py` still selects `lda_model20.pickle`, while this repository
contains `lda_model10.pickle`. Supply the intended 20-topic model or explicitly
select the intended model before running the full analysis.

Run the storage checks with `python -m unittest test_lda_storage`.
