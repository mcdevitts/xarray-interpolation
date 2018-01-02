import numpy as np
import unittest
import xarray as xr
import xinterp


da_1d_repeat = xr.DataArray(
    np.array((1, ), ),
    coords={'x': [0, ], },
    dims=('x', ),
)

da_1d_real = xr.DataArray(
    np.array([0, 1]),
    coords={'x': [0, 1], },
    dims=('x', ),
)

da_1d_complex = xr.DataArray(
    np.array((0, 1+1j), dtype='complex'),
    coords={'x': [0, 1], },
    dims=('x', ),
)


class TestCreation(unittest.TestCase):
    def test_1d_real(self):
        # Within bounds
        result = da_1d_real.interp.smart(x=[0, 0.25, 0.5, 0.75, 1])
        xr.testing.assert_equal(
            result,
            xr.DataArray(
                np.array([0, 0.25, 0.5, 0.75, 1]),
                coords={'x': [0, 0.25, 0.5, 0.75, 1]},
                dims=('x', ),
            )
        )
        # Outside bounds
        result = da_1d_real.interp.smart(x=[-0.1, 0.5, 1.1])
        xr.testing.assert_equal(
            result,
            xr.DataArray(
                np.array([-0.1, 0.5, 1.1]),
                coords={'x': [-0.1, 0.5, 1.1]},
                dims=('x',),
            )
        )

    def test_1d_complex(self):
        # Within bounds
        result = da_1d_complex.interp.smart(x=[0, 0.5, 1])
        print(result)
        xr.testing.assert_equal(
            result,
            xr.DataArray(
                np.array([0, 0.5 + 0.5j, 1 + 1j]),
                coords={'x': [0, 0.5, 1]},
                dims=('x', ),
            )
        )
        # Outside bounds
        result = da_1d_complex.interp.smart(x=[-0.1, 0.5, 1.1])
        xr.testing.assert_equal(
            result,
            xr.DataArray(
                np.array([-0.1 - 0.1j, 0.5 + 0.5j, 1.1 + 1.1j]),
                coords={'x': [-0.1, 0.5, 1.1]},
                dims=('x',),
            )
        )

    def test_1d_repeat(self):
        # Within bounds
        result = da_1d_repeat.interp.smart(x=[-1, 0, 1])
        xr.testing.assert_equal(
            result,
            xr.DataArray(
                np.array([1, 1, 1, ]),
                coords={'x': [-1, 0, 1]},
                dims=('x', ),
            )
        )
        # Outside bounds
        result = da_1d_repeat.interp.smart(x=[-2, 0, 2])
        xr.testing.assert_equal(
            result,
            xr.DataArray(
                np.array([1, 1, 1, ]),
                coords={'x': [-2, 0, 2]},
                dims=('x',),
            )
        )


