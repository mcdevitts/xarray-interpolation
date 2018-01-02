import numpy as np
import unittest
import xarray as xr
import xinterp

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


# TODO: The cases with only a single dimension provided should be moved to test_1d_interp!

class Test2DInterp(unittest.TestCase):
    def test_2d_real_within_bounds(self):
        result = da_2d_real.interp.smart(y=[0, 0.25, 0.5, 0.75, 1])
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

        result = da_2d_real.interp.smart(x=[0, 0.5, 1])
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

        result = da_2d_real.interp.smart(x=[0, 0.5, 1], y=[0, 0.25, 0.5, 0.75, 1])
        xr.testing.assert_equal(
                result,
                xr.DataArray(
                        np.array([[ 0.  ,  0.25,  0.5 ,  0.75,  1.  ],
                                  [ 0.5 ,  0.75,  1.  ,  1.25,  1.5 ],
                                  [ 1.  ,  1.25,  1.5 ,  1.75,  2.  ]]),
                        coords={
                            'x': [0, 0.5, 1],
                            'y': [0, 0.25, 0.5, 0.75, 1]
                        },
                        dims=('x', 'y'),
                )
        )

    def test_2d_real_outside_bounds(self):
        result = da_2d_real.interp.smart(y=[-0.1, 0.5, 1.1])
        xr.testing.assert_equal(
            result,
            xr.DataArray(
                np.array(([-0.1, 0.5, 1.1], [0.9, 1.5, 2.1])),
                coords={
                    'x': [0, 1],
                    'y': [-0.1, 0.5, 1.1]
                },
                dims=('x', 'y'),
            )
        )

    def test_2d_real_complex(self):
        pass

    def test_2d_extra_dimensions(self):
        result = da_2d_singular.interp.smart(x=[0, 1], y=[0, 0.5, 1])
        xr.testing.assert_equal(
            result,
            xr.DataArray(
                np.array(((0, 0.5, 1), (0, 0.5, 1))),
                coords={
                    'x': [0, 1],
                    'y': [0, 0.5, 1.0]
                },
                dims=('x', 'y'),
            )
        )




