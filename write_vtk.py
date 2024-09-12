import taichi as ti
import numpy as np
# from reservoir.points import Points
# https://vtk.org/Wiki/VTK/Writing_VTK_files_using_python
# https://github.com/taichi-dev/taichi/blob/560b740c46575e3bd0eb9bae3d24c9eca63376ff/python/taichi/tools/vtk.py

def write_scalar_field_vtk(scalar_field, filename):
    try:
        from pyevtk.hl import gridToVTK  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise RuntimeError(
            "Failed to import pyevtk. Please install it via /\
        `pip install pyevtk` first. "
        )

    scalar_field_np = scalar_field.to_numpy()
    field_shape = scalar_field_np.shape
    dimension = len(field_shape)

    if dimension not in (2, 3):
        raise ValueError("The input field must be a 2D or 3D scalar field.")

    if dimension == 2:
        scalar_field_np = scalar_field_np[np.newaxis, :, :]
        zcoords = np.array([0, 1])
    elif dimension == 3:
        zcoords = np.arange(0, field_shape[2])
    gridToVTK(
        filename,
        x=np.arange(0, field_shape[0]),
        y=np.arange(0, field_shape[1]),
        z=zcoords,
        cellData={filename: scalar_field_np},
    )



def write_particle_vtk(points, feature_dict, filename, verbose=False):
    try:
        from pyevtk.hl import pointsToVTK  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise RuntimeError(
            "Failed to import pyevtk. Please install it via /\
        `pip install pyevtk` first. "
        )

    # check dim
    dim=3
    p=np.zeros((dim))
    # if isinstance(points, Points):
    #     if points.dim not in (2, 3):
    #         raise ValueError("The input field must be a 2D or 3D scalar field.")
        
    #     if verbose:
    #         print("write_particle_vtk: parse points, dim=", points.dim, ", size=", points.x.shape)
    #     dim=points.dim
    #     p=points.x.to_numpy()
    if isinstance(points, ti.Field):
        p=points.to_numpy()
        dim=p.shape[-1]
        if verbose:
            print("write_particle_vtk: parse points from taichi, dim=", dim, ",size=", p.shape)
    
    if isinstance(points, np.ndarray):
        p=points
        dim=points.shape[-1]
        if verbose:
            print("write_particle_vtk: parse points from numpy, dim=", dim, ",size=", p.shape)
    

    # parse positions to numpy array
    x = np.ascontiguousarray(p[:,0])
    y = np.ascontiguousarray(p[:,1])
    z = np.zeros_like(y)
    if dim==3:
        z = np.ascontiguousarray(p[:,2])

    num=x.shape[0]
    if 'num' in feature_dict.keys():
        num=int(feature_dict['num'])
    
    x=x[0:num]
    y=y[0:num]
    z=z[0:num]

    # parse features to numpy array
    data={}
    for k,v in feature_dict.items():
        if k=='num':
            continue
        v_np=v
        if not isinstance(v_np, np.ndarray):
            v_np=v_np.to_numpy()
        v_np=v_np[0:num]
        result=v_np
        # write 3d vector
        if v_np.ndim==2 and v_np.shape[-1]==3:
            result=(np.ascontiguousarray(v_np[...,0]),np.ascontiguousarray(v_np[...,1]),np.ascontiguousarray(v_np[...,2]))
            if verbose:
                print("write_particle_vtk: parse vector feature, key=\"", k, "\", value shape=", v_np.shape)
        # write 2d vector
        elif v_np.ndim==2 and v_np.shape[-1]==2:
            result=(np.ascontiguousarray(v_np[...,0]),np.ascontiguousarray(v_np[...,1]),np.zeros_like(v_np[...,0]))
            if verbose:
                print("write_particle_vtk: parse vector feature, key=\"", k, "\", value shape=", v_np.shape)
        # write 2x2 tensor
        elif v_np.ndim==3 and v_np.shape[-2]==2 and v_np.shape[-1]==2:
            result=(np.ascontiguousarray(v_np[...,0,0]),np.ascontiguousarray(v_np[...,0,1]),np.zeros_like(v_np[...,0,0]))
            data[k+'x']=result
            result=(np.ascontiguousarray(v_np[...,1,0]),np.ascontiguousarray(v_np[...,1,1]),np.zeros_like(v_np[...,0,0]))
            data[k+'y']=result
            result=(np.zeros_like(v_np[...,0,0]),np.zeros_like(v_np[...,0,0]),np.ones_like(v_np[...,0,0]))
            data[k+'z']=result
            if verbose:
                print("write_particle_vtk: parse tensor feature, key=\"", k, "\", value shape=", v_np.shape)
        # write 3x3 tensor
        elif v_np.ndim==3 and v_np.shape[-2]==3 and v_np.shape[-1]==3:
            result=(np.ascontiguousarray(v_np[...,0,0]),np.ascontiguousarray(v_np[...,0,1]),np.ascontiguousarray(v_np[...,0,2]))
            data[k+'x']=result
            result=(np.ascontiguousarray(v_np[...,1,0]),np.ascontiguousarray(v_np[...,1,1]),np.ascontiguousarray(v_np[...,1,2]))
            data[k+'y']=result
            result=(np.ascontiguousarray(v_np[...,2,0]),np.ascontiguousarray(v_np[...,2,1]),np.ascontiguousarray(v_np[...,2,2]))
            data[k+'z']=result
            if verbose:
                print("write_particle_vtk: parse tensor feature, key=\"", k, "\", value shape=", v_np.shape)
        else:
            if verbose:
                print("write_particle_vtk: parse scalar feature, key=\"", k, "\", value shape=", v_np.shape)
        if  v_np.ndim!=3:
            data[k]=result

    # write out
    pointsToVTK(filename, x, y, z, data=data)

def write_particle_vtk_2(points, features, threshold, filename):
    try:
        from pyevtk.hl import pointsToVTK  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise RuntimeError(
            "Failed to import pyevtk. Please install it via /\
        `pip install pyevtk` first. "
        )

    points_pos_np = points.to_numpy()
    points_v_np = features.to_numpy()
    if points_pos_np.shape[1] not in (2, 3):
        raise ValueError("The input field must be a 2D or 3D scalar field.")

    x = np.array([])
    y = np.array([])
    z = np.array([])
    phi = np.array([])
    count = 0
    for i in range(0,points_pos_np.shape[0]):
        # change the range of phi to show more particles, now it is 1e-3 for reconstructing the surface of sphere
        if abs(points_v_np[i]) <= threshold:
            count += 1
            x = np.append(x, points_pos_np[i][0])
            y = np.append(y, points_pos_np[i][1])
            phi = np.append(phi, points_v_np[i])
            if points_pos_np.shape[1] == 3:
                # if dim == 3
                z = np.append(z, points_pos_np[i][2])
    # pressure = np.random.rand(npoints)
    # temp = np.random.rand(npoints)
    # pointsToVTK(filename, x, y, z, data = {"temp" : temp, "pressure" : pressure})
    if count > 0:
        pointsToVTK(filename, x, y, z, data = {"phi" : phi})

__all__ = ["write_vtk"]