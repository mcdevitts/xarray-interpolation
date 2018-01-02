
import copy
import numpy as np
import scipy
import scipy.interpolate
import xarray as xr


@xr.register_dataarray_accessor('interp')
class Interpolater(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def interp1d(self, bounds_error=False, fill_value=None, extend_dims=True, repeat=True, **vectors):
        """

        Parameters
        ----------
        bounds_error
        fill_value
        extend_dims
        vectors

        Returns
        -------
        data_array : DataArray

        """
        # There can and should only be one key
        k = list(vectors.keys())[0]
        xi = list(vectors.values())[0]

        # Determine which axis we want to interpolate on
        ax_idx = self._obj.dims.index(k)

        x = self._obj.coords[k]
        y = self._obj.data

        #
        if repeat and self._obj.shape[ax_idx] == 1:
            yi = np.repeat(y, len(xi), axis=ax_idx)

        else:
            # interp1d's extrapolate behavior is not enabled by default. Have to specify extrapolate in fill_value
            if not fill_value:
                fill_value = "extrapolate"

            # Smartly handle complex data
            if np.any(np.iscomplex(y)):
                f_real = scipy.interpolate.interp1d(x, np.real(y), axis=ax_idx, bounds_error=bounds_error,
                                                    fill_value=fill_value)
                f_imag = scipy.interpolate.interp1d(x, np.imag(y), axis=ax_idx, bounds_error=bounds_error,
                                                    fill_value=fill_value)
                yi = f_real(xi) + 1j * f_imag(xi)

            else:
                f = scipy.interpolate.interp1d(x, y, axis=ax_idx, bounds_error=bounds_error, fill_value=fill_value)
                yi = f(xi)

        # Build a new DataArray leveraging the previous coords object
        new_coords = copy.deepcopy(self._obj.coords)
        new_coords[k] = xi

        data_array = xr.DataArray(
                yi,
                coords=new_coords,
                dims=copy.deepcopy(self._obj.dims),
        )
        return data_array

    def interpn(self, bounds_error=False, fill_value=None, extend_dims=True, **vectors):
        """

        Parameters
        ----------
        bounds_error
        fill_value
        extend_dims
        vectors

        Returns
        -------
        data_array : DataArray

        """
        # Remove any singular dimensions within the data. These will be treated as extra, extended dimensions that
        # will be broadcast to.
        self._obj = self._obj.squeeze(drop=True)

        keys_interp = list(vectors.keys())
        keys_data = list(self._obj.dims)

        # Are the number of keys equal and are they the same keys?
        if set(keys_interp) == set(keys_data):
            # This is simple. Just interpolate the darn thing.
            data = copy.deepcopy(self._obj)
            i_data = self._interpn(data ,bounds_error=bounds_error, fill_value=fill_value, **vectors)
            data_array = xr.DataArray(i_data,
                                      coords=vectors,
                                      dims=vectors.keys())

        # Do keys_interp contain all the keys_data?
        elif set(keys_interp) > set(keys_data):
            # The user has requested we interpolate along more dimensions than
            # exists within the DataArray,
            if extend_dims:
                # Determine which dimensions need to be interpolated and which dimensions needs to be extended
                i_vectors = {k: v for k, v in vectors.items() if k in keys_data}
                i_keys = [k for k in vectors.keys() if k in keys_data]
                ext_vectors = {k: v for k, v in vectors.items() if k not in keys_data}
                ext_keys = [k for k in vectors.keys() if k not in keys_data]

                # Slicing of data is not necessary since all the dimensions are being interpolated
                data = copy.deepcopy(self._obj)
                i_data = self._interpn(data, bounds_error=bounds_error, fill_value=fill_value, **i_vectors)

                ext_vectors_shape = tuple(len(x) for x in ext_vectors.values())

                ext_data = np.broadcast_to(i_data, ext_vectors_shape + i_data.shape)
                data_array = xr.DataArray(ext_data,
                                          coords={**ext_vectors, **i_vectors},
                                          dims=ext_keys + i_keys)
                data_array = data_array.transpose(*vectors.keys())

            else:
                print('uh oh')
                raise NotImplementedError()

        # Do keys_data contain all the keys_interp?
        elif set(keys_interp) < set(keys_data):
            if extend_dims:
                # Interpolate all the dimensions within the data array
                ext_vectors = {k: v for k, v in vectors.items() if k not in keys_data}
                ext_keys = [k for k in vectors.keys() if k not in keys_data]

                dat = copy.deepcopy(self._obj)
                i_data = self._interpn(data, bounds_error=bounds_error, fill_value=fill_value, )

                # Extend the resulting data array to cover the additional vectors
                raise NotImplementedError()
            else:
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


    def smart(self, bounds_error=False, fill_value=None, repeat=True,
              extend_dims=True, **vectors):
        """Intelligently interpolate the xarray with multiple dimension and
        complex data.

        Automatically call ``scipy``'s interp1d or RegularGridInterpolator
        methods. This method also interpolates the real and complex as a
        superposition.

        Parameters
        ----------
        bounds_error: bool
            If True, when interpolated values are requested outside of the
            domain of the input data, a ValueError is raised. If False, then
            fill_value is used.
        fill_value: float, optional
            If provided, the value to use for points outside of the
            interpolation domain. If None, values outside the domain are
            extrapolated.
        repeat: bool
        expand_dims: bool

        vectors: dict of 1-D ndarrays
            Keys must match coords names.

        Returns
        -------
        data_array: xarray
            Interpolated array with dimensions matching those found in
            `vectors`.

        """
        # ensure every key is present
        # try:
        #     assert all([k in self._obj.coords.keys() for k in vectors])
        # except AssertionError as e:
        #     print(self._obj.coords.keys())
        #     print(vectors)
            # raise e

        # 1-D interpolation of an N-D structure

        if len(vectors) == 1:
            data_array = self.interp1d(bounds_error=bounds_error, fill_value=fill_value, extend_dims=extend_dims,
                                       repeat=True, **vectors)
        else:
            data_array = self.interpn(bounds_error=bounds_error, fill_value=fill_value, extend_dims=extend_dims,
                                      **vectors)

        return data_array


# da_1d_1 = xr.DataArray(
#     np.array((1, ), ),
#     coords={'x': [0, ], },
#     dims=('x', ),
# )
#
# da_1d = xr.DataArray(
#     np.random.randn(2, ),
#     coords={'x': [0, 1], },
#     dims=('x', )
# )
#
# da_2d = xr.DataArray(
#     np.random.randn(2, 3),
#     coords={'x': [0, 1], 'y': [1, 1.5, 2]},
#     dims=('x', 'y')
# )
#
# da_3d = xr.DataArray(
#     np.random.randn(2, 3, 4),
#     coords={'x': [0, 1], 'y': [1, 1.5, 2], 'z': [0, 1, 2, 3]},
#     dims=('x', 'y', 'z')
# )
#
#
# # Interpolates a 2-d matrix along one vector
# da_1d_i = da_1d.interp.smart(x=[0, 0.25, 0.5, 0.75, 1])
# da_1d_iv = da_1d.interp.smart(x=[0, 0.25, 0.5, 0.75, 1], y=[0, 1], extend_dims=True)
# da_1d_ii = da_1d_1.interp.smart(x=[-1, 0, 1])
# da_1d_iii = da_1d_1.interp.smart(x=[-1, 0, 1], y=[0, 1], extend_dims=True)
# # Interpolates a 2-d matrix in 2-d
# da_2d_i = da_2d.interp.smart(x=[0, 0.25, 0.5, 0.75, 1])
# da_2d_i2 = da_2d.interp.smart(x=[0, 0.25, 0.5, 0.75, 1], y=[1, 1.25, 1.75, 2])
# da_3d_i3 = da_3d.interp.smart(
#     x=[0, 0.25, 0.5, 0.75, 1, ],
#     y=[1, 1.25, 1.75, 2, ],
#     z=[1, 1.25, 1.75, 2, 2.25, 2.5, 2.75, 3, ],
# )
#
# # # Either... broadcast to this new dimension or just throw an error
# # # Throwing an error seems to make the most sense
# # da_i = da.interp.smart_interp(x=[0, 0.5], y=[1, 1.1, 1.2], z=[0,1])
#
# # Or...
#
# da_2d = xr.DataArray(
#     np.random.randn(2, 3),
#     coords={'x': [0, 1], 'y': [1, 1.5, 2]},
#     dims=('x', 'y')
# )
#
# # da_2d_i2 = da_2d.interp.smart(x=[0, 0.25, 0.5, 0.75, 1], y=[1, 1.25, 1.75, 2])
#
# c = da_2d.interp.interpn(y=[1, 1.5, 2], x=[0, 0.5, 1], z=[0, 1])
# d = da_2d.interp.interpn(x=[0, 0.5, 1], y=[1, 1.5, 2], z=[0, 1])
