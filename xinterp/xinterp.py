
import copy
import numpy as np
import scipy
import scipy.interpolate
import xarray as xr

@xr.register_dataarray_accessor('interp')
class Interpolater(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def smart_interp(self, **kwargs):
        """Create and return an interpolation function that smartly handles
        multiple dimensions and complex variables.

        Parameters
        ----------
        kwargs

        Returns
        -------
        f_interp: function

        """
        # kwargs is a set of k/v pairs of the coordinates to be interpolated upon.

        # ensure every key is present
        assert all([k in self._obj.coords.keys() for k in kwargs.keys()])

        if len(kwargs.keys()) == 1:
            # There can and should only be one key
            k = list(kwargs.keys())[0]
            xi = list(kwargs.values())[0]

            # Determine which axis we want to interpolate on
            ax_idx = list(self._obj.coords.keys()).index(k)

            x = self._obj.coords[k]
            y = self._obj.data

            # Smartly handle complex data
            if np.any(np.iscomplex(y)):
                f_real = scipy.interpolate.interp1d(x, np.real(y), axis=ax_idx)
                f_imag = scipy.interpolate.interp1d(x, np.imag(y), axis=ax_idx)
                yi = f_real(xi) + 1j * f_imag(xi)

            else:
                f = scipy.interpolate.interp1d(x, y, axis=ax_idx)
                yi = f(xi)

            # Build a new DataArray leveraging the previous coords object
            new_coords = copy.deepcopy(self._obj.coords)
            new_coords[k] = xi

            da = xr.DataArray(
                    yi,
                    coords=new_coords,
                    dims=list(new_coords.keys()),
            )
            return da

        else:
            # Must provide values for every axis present
            #assert self._obj.coords.keys()

            # Re-order kwargs into axis order
            keys = list(self._obj.coords.keys())
            points_original = list(self._obj.coords.values())
            points_interp = [kwargs[k] for k in self._obj.coords.keys()]
            output_shape = tuple(len(a) for a in points_interp)

            data = self._obj.data

            if np.any(np.iscomplex(data)):
                f_real = scipy.interpolate.RegularGridInterpolator(points_original, np.real(data))#, bounds_error=bounds_error, fill_value=fill_value)
                f_imag = scipy.interpolate.RegularGridInterpolator(points_original, np.imag(data))#, bounds_error=bounds_error, fill_value=fill_value)

                pts = np.reshape(np.meshgrid(*points_interp, indexing='ij'), (len(points_interp), np.prod(output_shape)))
                interp_data = f_real(pts.T) + 1j * f_imag(pts.T)
                return np.reshape(interp_data, output_shape)

            else:
                f = scipy.interpolate.RegularGridInterpolator(points_original, data) #, bounds_error=bounds_error, fill_value=fill_value)
                pts = np.reshape(np.meshgrid(*points_interp, indexing='ij'), (len(points_interp), np.prod(output_shape)))
                interp_data = f(pts.T)
                return np.reshape(interp_data, output_shape)


        # elif len(kwargs.keys()) == 2:

        #     # More interpolation variables than ndim. We can either:
        #     #   - Broadcast to the size and shape describe
        #     #   - Throw an error saying it's an invalid dimension *

        #     # Normal case
        #     if len(self._obj.coords.keys()) == 2:

        #         # Need to determine which variable is 'x' (1st) and 'y' (2nd)
        #         interp_vars = []
        #         for k in self._obj.coords.keys():
        #             interp_vars.append(kwargs[k])

        #         # Smartly handle complex data
        #         if np.any(np.iscomplex(self._obj.data)):
        #             f_real = scipy.interpolate.interp2d(*self._obj.coords.values(), np.real(self._obj.data).T)
        #             f_imag = scipy.interpolate.interp2d(*self._obj.coords.values(), np.imag(self._obj.data).T)
        #             zi = f_real(*interp_vars) + 1j * f_imag(*interp_vars).T

        #         else:
        #             f = scipy.interpolate.interp2d(*self._obj.coords.values(), self._obj.data.T)
        #             zi = f(*interp_vars).T

        #         new_coords = copy.deepcopy(self._obj.coords)
        #         for k, v in kwargs.items():
        #             new_coords[k] = v

        #         da = xr.DataArray(
        #             zi,
        #             coords=new_coords,
        #             dims=list(new_coords.keys()),
        #         )
        #         return da

        #     # Fewer dimensions than interpolation variables
        #     elif len(self._obj.coords.keys()) < 2:
        #         # Should never happen!
        #         raise NotImplementedError()

        #     # More dimensions than interpolation variables
        #     else:
        #         raise NotImplementedError("2D interpolation for >2D objects not supported currently.")
        # else:


    def _interp2d(self):
        pass

    def _interpnd(self):
        pass


da_1d = xr.DataArray(
        np.random.randn(2, ),
        coords={'x': [0, 1], },
        dims=('x', )
)

da_2d = xr.DataArray(
        np.random.randn(2, 3),
        coords={'x': [0, 1], 'y': [1, 1.5, 2]},
        dims=('x', 'y')
)

da_3d = xr.DataArray(
        np.random.randn(2, 3, 4),
        coords={'x': [0, 1], 'y': [1, 1.5, 2], 'z': [0, 1, 2, 3]},
        dims=('x', 'y', 'z')
)

# Interpolates a 2-d matrix along one vector
da_1d_i = da_1d.interp.smart_interp(x=[0, 0.25, 0.5, 0.75, 1])
# Interpolates a 2-d matrix in 2-d
da_2d_i = da_2d.interp.smart_interp(x=[0, 0.25, 0.5, 0.75, 1])
da_2d_i2 = da_2d.interp.smart_interp(x=[0, 0.25, 0.5, 0.75, 1], y=[1, 1.25, 1.75, 2])
da_3d_i3 = da_3d.interp.smart_interp(x=[0, 0.25, 0.5, 0.75, 1], y=[1, 1.25, 1.75, 2],)
                                     #z=[1, 1.25, 1.75, 2, 2.25, 2.5, 2.75, 3])

# # Either... broadcast to this new dimension or just throw an error
# # Throwing an error seems to make the most sense
# da_i = da.interp.smart_interp(x=[0, 0.5], y=[1, 1.1, 1.2], z=[0,1])

# Or...
