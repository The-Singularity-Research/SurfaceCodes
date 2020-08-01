from src.surface_code_graph import SurfaceCodeGraph
import pytest

def test_init():
    scg = SurfaceCodeGraph(sigma=[(0, 1, 2), (3, 4, 5), (6, 7, 8, 9)], alpha=[(0, 3), (1, 4), (2, 6), (5, 7), (8, 9)])
    


