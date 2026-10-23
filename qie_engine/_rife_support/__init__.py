"""RIFE's own `model/` package, vendored and inserted on sys.path at load time.

Kept as a subpackage of qie_engine so it is never installed under the bare name `model`; the
loader adds this directory to sys.path only while importing the downloaded RIFE archive.
"""
