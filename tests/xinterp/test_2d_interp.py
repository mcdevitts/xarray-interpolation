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


class Test2DInterp(unittest.TestCase):
    def test_2d_real_within_bounds(self):
        result = da_2d_real.interp.interpnd(x=[0, 0.5, 1], y=[0, 0.25, 0.5, 0.75, 1])
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
        # Extrapolate outside the bounds
        result = da_2d_real.interp.interpnd(x=[0, 1], y=[-0.1, 0.5, 1.1], bounds_error=False)
        xr.testing.assert_allclose(
            result,
            xr.DataArray(
                np.array([[-0.1, 0.5, 1.1],
                          [0.9, 1.5, 2.1]]),
                coords={
                    'x': [0, 1],
                    'y': [-0.1, 0.5, 1.1]
                },
                dims=('x', 'y'),
            )
        )
        # Fill value outside the bounds
        result = da_2d_real.interp.interpnd(x=[0, 1], y=[-0.1, 0.5, 1.1], bounds_error=False, fill_value=-100)
        xr.testing.assert_allclose(
            result,
            xr.DataArray(
                np.array([[-100, 0.5, -100],
                          [-100, 1.5, -100]]),
                coords={
                    'x': [0, 1],
                    'y': [-0.1, 0.5, 1.1]
                },
                dims=('x', 'y'),
            )
        )
        # Throw an error if the bounds are exceeded
        self.assertRaises(ValueError, da_2d_real.interp.interpnd, x=[0, 1], y=[-0.1, 0.5, 1.1], bounds_error=True)

    def test_2d_real_complex(self):
        pass

    def test_2d_extra_dimensions(self):
        result = da_2d_singular.interp.interpnd(x=[0, 1], y=[0, 0.5, 1], z=[0, 1])
        xr.testing.assert_equal(
            result,
            xr.DataArray(
                np.array([[[0., 0.],
                           [0.5, 0.5],
                           [1., 1.]],
                          [[0., 0.],
                           [0.5, 0.5],
                           [1., 1.]]]),
                coords={
                    'x': [0, 1],
                    'y': [0, 0.5, 1.0],
                    'z': [0, 1],
                },
                dims=('x', 'y', 'z'),
            )
        )

    def test_2d_underdimensioned(self):
        # TODO: Implemenet this case in xinterp!
        result = da_2d_real.interp.interpnd(x=[0, 1])
        xr.testing.assert_equal(
                result,
                xr.DataArray(
                    np.array(((0, 0.5, 1), (1, 1.5, 2))),
                    coords={
                        'x': [0, 1],
                        'y': [0, 0.5, 1]
                    },
                    dims=('x', 'y'),
                )
        )
