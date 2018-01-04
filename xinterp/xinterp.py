
"""

"""

import copy
import numpy as np
import scipy
import scipy.interpolate
import xarray as xr

__all__ = ('Interpolater', )


@xr.register_dataarray_accessor('interp')
class Interpolater(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def interp1d(self, bounds_error=False, fill_value=None, **vectors):
        """Interpolate the DataArray along a single dimension.

        Interpolation of N-D DataArrays along a single dimension is supported.

        If an axis is provided in `vectors` that does not match an already
        existing axis, the new axis will be added to the DataArray and the
        data tiled to fill it.

        Parameters
        ----------
        bounds_error: bool, optional
            If True, when interpolated values are requested outside of the
            domain of the input data, a ValueError is raised. If False, then
            fill_value is used.
        fill_value: bool, optional
            If provided, the value to use for points outside of the
            interpolation domain. If None, values outside the domain are
            extrapolated.
        vectors: dict of ndarray
            A dictionary containing a single interpolation vector. The vector
            must be a 1-D ndarray.

        Returns
        -------
        data_array : DataArray

        """

        # For now, always tile the array if a new dimension is provided!
        repeat = True

        # There should only be a single interpolation vector!
        assert len(vectors) == 1, "Only a single interpolation vector can be provided to interp1d!"

        # Create a local copy to ensure the original DataArray is not modified in anyway
        da = copy.deepcopy(self._obj)

        # Fetch the first (and only) interpolation vector
        k, xi = list(vectors.items())[0]

        # Determine which axis we want to interpolate on
        try:
            ax_idx = da.dims.index(k)
            x = da.coords[k]
            y = da.data
        except IndexError:
            raise IndexError("Invalid vector name: {0}. Name must correspond with one of the DataArray's axes.")

        if repeat and da.shape[ax_idx] == 1:
            yi = np.repeat(y, len(xi), axis=ax_idx)

        else:
            # interp1d's extrapolate behavior is not enabled by default. Have to specify extrapolate in fill_value
            if not fill_value and not bounds_error:
                fill_value = "extrapolate"

            # If the data is complex, interpolate the superposition of the real and imaginary parts
            if np.any(np.iscomplex(y)):
                f_real = scipy.interpolate.interp1d(x, np.real(y), axis=ax_idx, bounds_error=bounds_error,
                                                    fill_value=fill_value)
                f_imag = scipy.interpolate.interp1d(x, np.imag(y), axis=ax_idx, bounds_error=bounds_error,
                                                    fill_value=fill_value)
                yi = f_real(xi) + 1j * f_imag(xi)

            # Otherwise, just interpolate as usual
            else:
                f = scipy.interpolate.interp1d(x, y, axis=ax_idx, bounds_error=bounds_error, fill_value=fill_value)
                yi = f(xi)

        # Build a new DataArray leveraging the previous coords object
        new_coords = copy.deepcopy(da.coords)
        new_coords[k] = xi

        data_array = xr.DataArray(
            yi,
            coords=new_coords,
            dims=copy.deepcopy(da.dims),
        )
        return data_array

    def interpnd(self, bounds_error=False, fill_value=None, **vectors):
        """Interpolate a N-D DataArray along multiple dimensions.

        If an axis is provided in `vectors` that does not match an already
        existing axis, the new axis will be added to the DataArray and the
        data tiled to fill it.

        Parameters
        ----------
        bounds_error: bool, optional
            If True, when interpolated values are requested outside of the
            domain of the input data, a ValueError is raised. If False, then
            fill_value is used.
        fill_value: bool, optional
            If provided, the value to use for points outside of the
            interpolation domain. If None, values outside the domain are
            extrapolated.
        vectors: dict of ndarrays
            A dictionary containing interpolation vectors. The vectors
            must be 1-D ndarrays.

        Returns
        -------
        data_array : DataArray

        """

        # Ensure all vectors are str, ndarray pairs.
        for k, v in vectors.items():
            if not isinstance(k, str):
                raise TypeError('Invalid vector key: {0}! Key must be of type str.'.format(k))
            if not isinstance(v, (np.ndarray, list, tuple)):
                raise TypeError('Invalid vector for key: {0}! Vector must be of type ndarray.'.format(k))

        # Remove any singular dimensions within the data. These will be treated as extra, extended dimensions that
        # will be broadcast to.
        # Create a local copy of the array so that any modifications do not impact the original
        da = self._obj.squeeze(drop=True)

        keys_interp = list(vectors.keys())
        keys_data = list(da.dims)

        # Does the data have any keys? If not, just broadcast the value to the desired interpolation vectors
        if not keys_data:
            data = copy.copy(da.data)
            vectors_shape = tuple(len(x) for x in vectors.values())

            ext_data = np.broadcast_to(data, vectors_shape)

            data_array = xr.DataArray(ext_data,
                                      coords=vectors,
                                      dims=vectors.keys())
            data_array = data_array.transpose(*vectors.keys())

        # Are the number of keys equal and are they the same keys?
        elif set(keys_interp) == set(keys_data):
            # This is simple. Just interpolate the darn thing.
            data = copy.deepcopy(da)
            i_data = self._interpn(data ,bounds_error=bounds_error, fill_value=fill_value, **vectors)
            data_array = xr.DataArray(i_data,
                                      coords=vectors,
                                      dims=vectors.keys())

        # Do keys_interp contain all the keys_data?
        elif set(keys_interp) > set(keys_data):
            # Determine which dimensions need to be interpolated and which dimensions needs to be extended
            i_vectors = {k: v for k, v in vectors.items() if k in keys_data}
            i_keys = [k for k in vectors.keys() if k in keys_data]
            ext_vectors = {k: v for k, v in vectors.items() if k not in keys_data}
            ext_keys = [k for k in vectors.keys() if k not in keys_data]

            # Slicing of data is not necessary since all the dimensions are being interpolated
            data = copy.deepcopy(da)
            i_data = self._interpn(data, bounds_error=bounds_error, fill_value=fill_value, **i_vectors)

            ext_vectors_shape = tuple(len(x) for x in ext_vectors.values())

            ext_data = np.broadcast_to(i_data, ext_vectors_shape + i_data.shape)

            data_array = xr.DataArray(ext_data,
                                      coords={**ext_vectors, **i_vectors},
                                      dims=ext_keys + i_keys)
            data_array = data_array.transpose(*vectors.keys())


        # Do keys_data contain all the keys_interp?
        elif set(keys_interp) < set(keys_data):
            raise NotImplementedError()

        return data_array

    def _interpn(self, da, bounds_error=False, fill_value=None, **vectors):
        # Re-order vectors into axis order
        points_original = list(da.coords.values())
        points_interp = [vectors[k] for k in da.dims]
        output_shape = tuple(len(a) for a in points_interp)

        if np.any(np.iscomplex(da)):
            f_real = scipy.interpolate.RegularGridInterpolator(points_original, np.real(da.data),
                                                               bounds_error=bounds_error, fill_value=fill_value)
            f_imag = scipy.interpolate.RegularGridInterpolator(points_original, np.imag(da.data),
                                                               bounds_error=bounds_error, fill_value=fill_value)
            pts = np.reshape(np.meshgrid(*points_interp, indexing='ij'), (len(points_interp), np.prod(output_shape)))
            interp_data = f_real(pts.T) + 1j * f_imag(pts.T)

        else:
            f = scipy.interpolate.RegularGridInterpolator(points_original, da.data,
                                                          bounds_error=bounds_error, fill_value=fill_value)

            pts = np.reshape(np.meshgrid(*points_interp, indexing='ij'), (len(points_interp), np.prod(output_shape)))
            interp_data = f(pts.T)
        return np.reshape(interp_data, output_shape)


    def smart(self, bounds_error=False, fill_value=None, **vectors):
        """Intelligently interpolate the xarray with multiple dimension and
        complex data.

        Automatically call ``scipy``'s interp1d or RegularGridInterpolator
        methods. This method also interpolates the real and complex as a
        superposition.

        Parameters
        ----------
        bounds_error: bool, optional
            If True, when interpolated values are requested outside of the
            domain of the input data, a ValueError is raised. If False, then
            fill_value is used.
        fill_value: float, optional
            If provided, the value to use for points outside of the
            interpolation domain. If None, values outside the domain are
            extrapolated.

        vectors: dict of 1-D ndarrays
            A dictionary containing interpolation vectors. The vectors
            must be 1-D ndarrays.

        Returns
        -------
        data_array: xarray

        """
        if len(vectors) == 1:
            data_array = self.interp1d(bounds_error=bounds_error, fill_value=fill_value, **vectors)
        else:
            data_array = self.interpnd(bounds_error=bounds_error, fill_value=fill_value, **vectors)
        return data_array
