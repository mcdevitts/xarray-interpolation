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

da_2d_real = xr.DataArray(
    np.array(((0, 0.5, 1), (1, 1.5, 2))),
    coords={'x': [0, 1], 'y': [0, 0.5, 1]},
    dims=('x', 'y')
)

da_2d_singular = xr.DataArray(
    np.array(((0, 0.5, 1), )),
    coords={'x': [0, ], 'y': [0, 0.5, 1]},
    dims=('x', 'y')
)


class Test1DInterp(unittest.TestCase):
    def test_1d_real_within_bounds(self):
        # Within bounds
        result = da_1d_real.interp.interp1d(x=[0, 0.25, 0.5, 0.75, 1])
        xr.testing.assert_equal(
            result,
            xr.DataArray(
                np.array([0, 0.25, 0.5, 0.75, 1]),
                coords={'x': [0, 0.25, 0.5, 0.75, 1]},
                dims=('x', ),
            )
        )

    def test_1d_real_outside_bounds(self):
        # Extrapolation
        result = da_1d_real.interp.interp1d(x=[-0.1, 0.5, 1.1], bounds_error=False)
        xr.testing.assert_equal(
            result,
            xr.DataArray(
                np.array([-0.1, 0.5, 1.1]),
                coords={'x': [-0.1, 0.5, 1.1]},
                dims=('x',),
            )
        )
        # Fill value outside of bounds
        result = da_1d_real.interp.interp1d(x=[-0.1, 0.5, 1.1], bounds_error=False, fill_value=-100)
        xr.testing.assert_equal(
                result,
                xr.DataArray(
                        np.array([-100, 0.5, -100]),
                        coords={'x': [-0.1, 0.5, 1.1]},
                        dims=('x',),
                )
        )
        # Illegal
        self.assertRaises(ValueError, da_1d_real.interp.interp1d, x=[-0.1, 0.5, 1.1], bounds_error=True)


    def test_1d_complex_within_bounds(self):
        result = da_1d_complex.interp.interp1d(x=[0, 0.5, 1])
        xr.testing.assert_equal(
            result,
            xr.DataArray(
                np.array([0, 0.5 + 0.5j, 1 + 1j]),
                coords={'x': [0, 0.5, 1]},
                dims=('x', ),
            )
        )
    def test_1d_complex_outside_bounds(self):
        result = da_1d_complex.interp.interp1d(x=[-0.1, 0.5, 1.1])
        xr.testing.assert_equal(
            result,
            xr.DataArray(
                np.array([-0.1 - 0.1j, 0.5 + 0.5j, 1.1 + 1.1j]),
                coords={'x': [-0.1, 0.5, 1.1]},
                dims=('x',),
            )
        )

    def test_1d_repeat(self):
        result = da_1d_repeat.interp.interp1d(x=[-1, 0, 1])
        xr.testing.assert_equal(
            result,
            xr.DataArray(
                np.array([1, 1, 1, ]),
                coords={'x': [-1, 0, 1]},
                dims=('x', ),
            )
        )


    def test_2d_1d(self):
        result = da_2d_real.interp.interp1d(y=[0, 0.25, 0.5, 0.75, 1])
        xr.testing.assert_equal(
            result,
            xr.DataArray(
                np.array(([0, 0.25, 0.5, 0.75, 1], [1, 1.25, 1.5, 1.75, 2])),
                coords={
                    'x': [0, 1],
                    'y': [0, 0.25, 0.5, 0.75, 1]
                },
                dims=('x', 'y'),
            )
        )

        result = da_2d_real.interp.interp1d(x=[0, 0.5, 1])
        xr.testing.assert_equal(
            result,
            xr.DataArray(
                np.array(([0, 0.5, 1], [0.5, 1.0, 1.5],  [1, 1.5, 2])),
                coords={
                    'x': [0, 0.5, 1],
                    'y': [0, 0.5, 1]
                },
                dims=('x', 'y'),
            )
        )
