import sys
sys.path.insert(0, '/Users/dpn/proj/bout++/grid/xBOUT')
import xbout as xb
import xarray as xr
import numpy as np
import cmath, math
import vtk
from vtk.util import numpy_support
import copy
import adios2 as ad


def dumpGrid(ll, ul, lr, ur, var) :
    nx = 68
    ny = 64
    grid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    grid.SetPoints(points)

    ptId = 0
    for i in range(nx) :
        for j in range(ny) :
            p0 = (ll[0][i,j], ll[1][i,j], 0)
            p1 = (lr[0][i,j], lr[1][i,j], 0)
            p2 = (ur[0][i,j], ur[1][i,j], 0)
            p3 = (ul[0][i,j], ul[1][i,j], 0)
            points.InsertNextPoint(p0)
            points.InsertNextPoint(p1)
            points.InsertNextPoint(p2)
            points.InsertNextPoint(p3)
            grid.InsertNextCell(vtk.VTK_QUAD, 4, [ptId+0, ptId+1, ptId+2, ptId+3])
            ptId = ptId + 4

    writer = vtk.vtkDataSetWriter()
    writer.SetFileName('/Users/dpn/reader.vtk')
    writer.SetInputData(grid)
    writer.Write()


def matMul(a,b) :
    ar = a.real.values
    d = a.dims
    br = b.real.values
    c = ar * br
    cc = xr.DataArray(c, dims=b.dims)
    return cc

    c = np.zeros(a.shape, dtype=ar.dtype)
    for i in range(a.shape[0]) :
        for j in range(a.shape[1]) :
            for k in range(a.shape[2]) :
                c[i,j,k] = ar[i,j,k] * br[i,j,k]
    return c


def doFFT(ds, varName) :

    fft = np.fft

    nz = ds.metadata['nz']
    var = ds[varName].values
    dy = ds['dy'].values
    zShift = ds['zShift'] #.values
    axis = 2 #zeta

    zlength = nz * ds['dz'].values.flatten()[0]
    nmodes = nz // 2 + 1

    print('dims= ', ds.dims)
    print('zdim= ', ds.metadata['bout_zdim'])
    print('zlength= ', zlength, ' nmodes= ', nmodes, ' axis= ', axis)
    print('nz= ', nz)
    print('zShift= ', zShift.shape)
    print('phi= ', var.shape)
    print('dy= ', dy.shape)

    data_fft = fft.rfft(var, axis=axis)
    data_fft = xr.DataArray(data_fft, dims=('x', 'theta', 'zeta'))
    print('var.shape, data_fft.shape=', var.shape, data_fft.shape)
    print('axis= ', axis)

    kz = 2.0 * np.pi * xr.DataArray(np.arange(0, nmodes), dims="kz") / zlength
    print('kz= ', kz.shape)

    phase = 1.0j * kz * zShift
    print('phase= ', phase.shape)
    phase = phase.transpose('x', 'theta', 'kz', transpose_coords=True)
    #phase = np.transpose(phase, [1,2,0])
    print('phase= ', phase.shape)
    exp_phase = np.exp(phase)
    print('exp_phase.shape= ', exp_phase.shape)
    print('data_fft.shape= ', data_fft.shape)
    #data_shifted_fft = data_fft * np.exp(phase)
    data_shifted_fft = matMul(data_fft, exp_phase)
    #data_shifted_fft = np.multiply(data_fft, exp_phase)
    print('data_shifted_fft.shape= ', data_shifted_fft.shape)

    data_shifted = fft.irfft(data_shifted_fft, n=nz, axis=axis)
    print('data_shifted.shape= ', data_shifted.shape)
    return data_shifted

def getPoints(ll, lr, ur, ul, i, j) :
    p0 = (ll[0][i,j], ll[1][i,j],0)
    p1 = (lr[0][i,j], lr[1][i,j],0)
    p2 = (ur[0][i,j], ur[1][i,j],0)
    p3 = (ul[0][i,j], ul[1][i,j],0)
    return (p0,p3,p2,p1)

def dumpVTK(ds, fname) :
    nx = ds.shape[0]
    ny = ds.shape[1]
    #nx = 10
    #ny = 10
    grid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    grid.SetPoints(points)

    ll = (ds.Rxy_lower_left_corners.values, ds.Zxy_lower_left_corners.values)
    lr = (ds.Rxy_lower_right_corners.values, ds.Zxy_lower_right_corners.values)
    ul = (ds.Rxy_upper_left_corners.values, ds.Zxy_upper_left_corners.values)
    ur = (ds.Rxy_upper_right_corners.values, ds.Zxy_upper_right_corners.values)

    var = vtk.vtkFloatArray()
    var.SetName('phi')
    grid.GetCellData().AddArray(var)
    vv = ds.values[0,0]

    ptId = 0
    for i in range(nx) :
        for j in range(ny) :
            (p0,p1,p2,p3) = getPoints(ll, lr, ur, ul, i,j)
            points.InsertNextPoint(p0)
            points.InsertNextPoint(p1)
            points.InsertNextPoint(p2)
            points.InsertNextPoint(p3)
            if False :
                grid.InsertNextCell(vtk.VTK_VERTEX, 1, [ptId+0])
                grid.InsertNextCell(vtk.VTK_VERTEX, 1, [ptId+1])
                grid.InsertNextCell(vtk.VTK_VERTEX, 1, [ptId+2])
                grid.InsertNextCell(vtk.VTK_VERTEX, 1, [ptId+3])
            else :
                ptIds = [ptId+0,ptId+1,ptId+2,ptId+3]
                grid.InsertNextCell(vtk.VTK_QUAD, 4, ptIds)
            ptId = ptId + 4

            var.InsertNextValue(ds.values[i,j,0])

    writer = vtk.vtkDataSetWriter()
    writer.SetFileVersion(42)
    writer.SetFileName('/Users/dpn/core.vtk')
    writer.SetInputData(grid)
    writer.Write()




def dumpRegion(ds) :
    dsHigh = ds['phi'].bout.interpolate_parallel(n=2)
    #dsHigh = ds['phi']

    #dsHigh = dsHigh.isel(zeta=0)
    #nX,nY,nZ = (50,50,50)
    #cartesian = dsHigh.bout.interpolate_to_cartesian(nX, nY, nZ)

    core = dsHigh.bout.from_region('core')
    dumpVTK(dsHigh, '/Users/dpn/core.vtk')
    print('all done...')


def getPts(ll,lr,ur,ul,zShift,dz, i,j, k) :
    #dPhi = zeta[k+1]-zeta[k]
    #dPhi = 0.2

    points = []
    #Z = [k*dPhi, (k+1) * dPhi]
    #Z = [zeta[k], zeta[k+1]]
    Z = [k*0.05, (k+1)*.05] #k,k+1]
    Z = [0]
    for n in range(len(Z)) :
        print('ijk: ', (i,j,k), Z[n], 'zs: ', zShift[i,j])
        z0 = zShift[i,j] + Z[n]
        z1 = zShift[i,j] + Z[n]
        z2 = zShift[i,j] + Z[n]
        z3 = zShift[i,j] + Z[n]
        z0 = Z[n]
        z1 = Z[n]
        z2 = Z[n]
        z3 = Z[n]
        cylPts = []
        cylPts.append((ll[0][i,j], ll[1][i,j], z0))
        cylPts.append((lr[0][i,j], lr[1][i,j], z1))
        cylPts.append((ur[0][i,j], ur[1][i,j], z2))
        cylPts.append((ul[0][i,j], ul[1][i,j], z3))

        doXYZ = True
        for p in cylPts :
            if doXYZ :
                _r = p[0]
                _z = p[1]
                _theta = p[2]
                x = _r*math.cos(_theta)
                y = _r*math.sin(_theta)
                z = _z
                points.append((x,y,z))
            else :
                points.append(p)


    return points

def getCornerPts(ll,lr,ur,ul,zShift, i,j) :

    cylPts = []
    cylPts.append((ll[0][i,j], ll[1][i,j], 0))
    cylPts.append((lr[0][i,j], lr[1][i,j], 0))
    cylPts.append((ur[0][i,j], ur[1][i,j], 0))
    cylPts.append((ul[0][i,j], ul[1][i,j], 0))

    return cylPts

def writeVTK(grid, fname ) :
    writer = vtk.vtkDataSetWriter()
    writer.SetFileVersion(42)
    writer.SetFileName(fname)
    writer.SetInputData(grid)
    writer.Write()


def dumpQuadMesh(ptList, zShift) :
    grid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    grid.SetPoints(points)

    doQuad = False
    ptId = 0
    for p in ptList :
        points.InsertNextPoint(p)
        if doQuad :
            if (ptId+1) % 4 == 0 :
                ptIds = [ptId-3, ptId-2, ptId-1, ptId-0]
                grid.InsertNextCell(vtk.VTK_QUAD, 4, ptIds)
        else :
            if (ptId+1) % 8 == 0 :
                ptIds = [ptId-7, ptId-6, ptId-5, ptId-4, ptId-3, ptId-2, ptId-1, ptId-0]
                grid.InsertNextCell(vtk.VTK_HEXAHEDRON, 8, ptIds)
        ptId = ptId+1

    if doQuad :
        var = vtk.vtkFloatArray()
        var.SetName('zShift')
        for i in range(68-1) :
            for j in range(64-1) :
                var.InsertNextValue(zShift[i,j])
        #var.DeepCopy(numpy_support.numpy_to_vtk(zShift))
        #grid.GetCellData().AddArray(var)

    writeVTK(grid, '/Users/dpn/reader.vtk')

def CellToPt(grid, varNames) :
    recenter = vtk.vtkCellDataToPointData()
    recenter.SetInputData(grid)
    recenter.Update()
    grid2 = recenter.GetOutput()
    for v in varNames :
        arr = grid2.GetPointData().GetArray(v)
        v2 = v + '_pt'
        arr.SetName(v2)
        grid.GetPointData().AddArray(arr)

    return grid

def addZShiftStuff(grid, data, nx, ny, guardJ) :
    zShift = data.read('zShift')

    zShiftPt = grid.GetPointData().GetArray('zShift_pt')
    zShiftDiff = vtk.vtkFloatArray()
    zShiftDiff.SetNumberOfValues(grid.GetNumberOfPoints())
    zShiftDiff.SetName('zShiftDiff')
    grid.GetPointData().AddArray(zShiftDiff)
    zShiftIDiff = vtk.vtkFloatArray()
    zShiftIDiff.SetNumberOfValues(grid.GetNumberOfPoints())
    zShiftIDiff.SetName('zShift_i_Diff')
    grid.GetPointData().AddArray(zShiftIDiff)

    nCells = grid.GetNumberOfCells()
    for cellIdx in range(nCells) :
        cell = grid.GetCell(cellIdx)
        ptIds = cell.GetPointIds()
        n = ptIds.GetNumberOfIds()
        ids = [ptIds.GetId(0), ptIds.GetId(1), ptIds.GetId(2), ptIds.GetId(3)]

        i = cellIdx // ny
        j = cellIdx % ny
        j = j + guardJ
        zShiftCell = zShift[i,j]

        prevZShift = zShiftCell
        nextZShift = zShiftCell
        i_p1ZShift = zShiftCell
        i_m1ZShift = zShiftCell
        if j > 0 : prevZShift = zShift[i,j-1]
        if j < ny-1 : nextZShift = zShift[i,j+1]
        if i > 0 : i_m1ZShift = zShift[i-1,j]
        if i < nx-1 : i_p1ZShift = zShift[i+1,j]

        j01ZShift = abs(zShiftCell - nextZShift)
        j23ZShift = abs(zShiftCell - prevZShift)
        i12ZShift = abs(zShiftCell - i_p1ZShift)
        i03ZShift = abs(zShiftCell - i_m1ZShift)

        zShiftDiff.SetValue(ids[0], j01ZShift)
        zShiftDiff.SetValue(ids[1], j01ZShift)
        zShiftDiff.SetValue(ids[2], j23ZShift)
        zShiftDiff.SetValue(ids[3], j23ZShift)

        zShiftIDiff.SetValue(ids[0], i03ZShift)
        zShiftIDiff.SetValue(ids[1], i12ZShift)
        zShiftIDiff.SetValue(ids[2], i12ZShift)
        zShiftIDiff.SetValue(ids[3], i03ZShift)

    writeVTK(grid, '/Users/dpn/quad3D.vtk')
    return grid

def getHexPoints(quadPoints, k, dz, dz01, dz23, numRefine) :

    def interpolate(xRange, t ) :
        dx = xRange[1] - xRange[0]
        xi = xRange[0] + t*dx
        return xi

    result = []

    ## first plane.
    #kzShift = k*(dPhi+dz01+dz23)
    #k_z01 = kzShift + dz01
    #k_z23 = kzShift + dz23

    k_z01 = k*dz + dz01
    k_z23 = k*dz + dz23
    ## last plane
    #k1zShift = (k+1) * (dPhi+dz01+dz23)
    #k1_z01 = k1zShift + dz01
    #k1_z23 = k1zShift + dz23

    k1_z01 = (k+1) * dz + dz01
    k1_z23 = (k+1) * dz + dz23

    # add first plane
    p0 = copy.deepcopy(quadPoints[0])
    p1 = copy.deepcopy(quadPoints[1])
    p2 = copy.deepcopy(quadPoints[2])
    p3 = copy.deepcopy(quadPoints[3])
    p0[2] = k_z01
    p1[2] = k_z01
    p2[2] = k_z23
    p3[2] = k_z23
    result.append((p0,p1,p2,p3))

    #interpolate between first and last planes if numRefine specified.
    if numRefine > 0 :
        z01Range = (k_z01, k1_z01)
        z23Range = (k_z23, k1_z23)
        dT = 1.0 / float(numRefine+1)
        t = dT
        for n in range(numRefine) :
            pi = copy.deepcopy(quadPoints)
            zt = interpolate(z01Range, t)
            _zi = pi[0][2] + zt
            pi[0][2] = _zi
            pi[1][2] = _zi

            zt = interpolate(z23Range, t)
            _zi = pi[2][2] + zt
            pi[2][2] = _zi
            pi[3][2] = _zi
            result.append((pi[0], pi[1], pi[2], pi[3]))
            t = t + dT

    # add last plane
    p0 = (quadPoints[0][0], quadPoints[0][1], k1_z01)
    p1 = (quadPoints[1][0], quadPoints[1][1], k1_z01)
    p2 = (quadPoints[2][0], quadPoints[2][1], k1_z23)
    p3 = (quadPoints[3][0], quadPoints[3][1], k1_z23)
    result.append((p0,p1,p2,p3))

    return result

def build3DMesh(grid2D, data, nx, ny, guardJ) :
    zShiftMap = {}
    def getPt(_ds, idx) :
        _p = _ds.GetPoint(idx)
        return [_p[0], _p[1], 0]
    def inBounds(i,j, bounds) :
        if i < bounds[0][0] or i > bounds[0][1] :
            return False
        if j < bounds[1][0] or j > bounds[1][1] :
            return False
        return True

    zShift = data.read('zShift')
    phi = data.read('phi')
    nk = phi.shape[2]

    grid3D = vtk.vtkUnstructuredGrid()
    pts = vtk.vtkPoints()
    grid3D.SetPoints(pts)
    arrI = vtk.vtkFloatArray()
    arrJ = vtk.vtkFloatArray()
    arrK = vtk.vtkFloatArray()
    arrZShift = vtk.vtkFloatArray()
    arrI.SetName('I')
    arrJ.SetName('J')
    arrK.SetName('K')
    arrZShift.SetName('zShift')

    grid3D.GetCellData().AddArray(arrI)
    grid3D.GetCellData().AddArray(arrJ)
    grid3D.GetCellData().AddArray(arrK)
    grid3D.GetCellData().AddArray(arrZShift)

    #zShift = ds.GetPointData().GetArray('zShiftDiff')
    zShift = grid2D.GetCellData().GetArray('zShift')
    np = grid2D.GetNumberOfPoints()
    nc = grid2D.GetNumberOfCells()

    ptId = 0
    coreBounds = [(0,32), (12,52)]
    coreBounds = [(5,25), (13,49)]
    coreBounds = [(25,25), (13,49)]
    #coreBounds = [(25,25), (15,15)]

    dz = 0.015514
    nk = 81
    for k in range(nk) :
        for cellIdx in range(nc) :
            i = cellIdx // ny
            j = cellIdx % ny

            if not inBounds(i,j, coreBounds) :
                continue

            ##skip the corner cases....
            if j == 0 or j == ny-1 : continue

            if i == 25 and j in [17,18,19,20] :
                print('pause here to look at zShift')


            cell = grid2D.GetCell(cellIdx)
            ptIds = cell.GetPointIds()

            ids = [ptIds.GetId(0), ptIds.GetId(1), ptIds.GetId(2), ptIds.GetId(3)]
            quadPts = (getPt(grid2D,ids[0]), getPt(grid2D,ids[1]), getPt(grid2D,ids[2]), getPt(grid2D,ids[3]))

            zij = zShift.GetValue(cellIdx)
            arrZShift.InsertNextValue(zij)
            ij_p1 = i*ny + (j+1)
            ij_m1 = i*ny + (j-1)
            zij1 = zShift.GetValue(ij_p1)
            zij_1 = zShift.GetValue(ij_m1)

            dz01 = abs(zij1 - zij)
            dz23 = abs(zij_1 - zij)

            ##John's suggestion:
            dz01 = 0.5*(zij1 - zij)
            dz23 = 0.5*(zij_1-zij)
            dzFactor = 1
            #dzFactor = 1.0
            dz01 = dz01 * dzFactor
            dz23 = dz23 * dzFactor

            numRefine = 0
            hexPoints = getHexPoints(quadPts, k, dz, dz01, dz23, numRefine)

            # add the points and cells for each hex.
            for cnt in range(len(hexPoints)-1) :
                pln0 = hexPoints[cnt]
                pln1 = hexPoints[cnt+1]
                for _p in pln0 : pts.InsertNextPoint(_p)
                for _p in pln1 : pts.InsertNextPoint(_p)

                grid3D.InsertNextCell(vtk.VTK_HEXAHEDRON, 8, [ptId+0, ptId+1, ptId+2, ptId+3, ptId+4, ptId+5, ptId+6, ptId+7])
                arrI.InsertNextValue(i)
                arrJ.InsertNextValue(j)
                arrK.InsertNextValue(k)

                ptId = ptId+8
                if k == nk-1 :
                    print(_p[2])


    writeVTK(grid3D, '/Users/dpn/grid3D.vtk')
    return grid3D


def buildQuadMesh(ds, data, guardJ) :
    #ll = (ds['Rxy_lower_left_corners'].values, ds['Zxy_lower_left_corners'].values)
    ll = (ds['Rxy_corners'].values, ds['Zxy_corners'].values)
    lr = (ds['Rxy_lower_right_corners'].values, ds['Zxy_lower_right_corners'].values)
    ur = (ds['Rxy_upper_right_corners'].values, ds['Zxy_upper_right_corners'].values)
    ul = (ds['Rxy_upper_left_corners'].values, ds['Zxy_upper_left_corners'].values)
    phi = data.read('phi')
    zShift = data.read('zShift')

    nx = ll[0].shape[0]
    ny = ll[0].shape[1]

    i_range = range(nx)
    j_range = range(ny)

    grid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    grid.SetPoints(points)
    phiVar, zShiftVar, cellIdVar, ptIdVar = vtk.vtkFloatArray(), vtk.vtkFloatArray(), vtk.vtkFloatArray(), vtk.vtkFloatArray()
    phiVar.SetName('phi')
    zShiftVar.SetName('zShift')
    ptIdVar.SetName('pointId')
    cellIdVar.SetName('cellId')
    grid.GetCellData().AddArray(phiVar)
    grid.GetCellData().AddArray(zShiftVar)
    grid.GetCellData().AddArray(cellIdVar)
    grid.GetPointData().AddArray(ptIdVar)
    idxIVar, idxJVar = vtk.vtkFloatArray(), vtk.vtkFloatArray()
    idxIVar.SetName('i')
    idxJVar.SetName('j')
    grid.GetCellData().AddArray(idxIVar)
    grid.GetCellData().AddArray(idxJVar)

    ptId = 0
    cellId = 0
    for i in i_range :
        for j in j_range :
            pts = getCornerPts(ll,lr,ur,ul,zShift, i,j)
            for p in pts :
                points.InsertNextPoint(p)
                ptIdVar.InsertNextValue(ptId)
                ptId = ptId+1

            grid.InsertNextCell(vtk.VTK_QUAD, 4, [ptId-4, ptId-3, ptId-2, ptId-1])
            phiVar.InsertNextValue(phi[i,j+guardJ,0])
            zShiftVar.InsertNextValue(zShift[i,j+guardJ])
            cellIdVar.InsertNextValue(cellId)
            idxIVar.InsertNextValue(i)
            idxJVar.InsertNextValue(j)
            cellId = cellId+1

    grid = CellToPt(grid, ['phi', 'zShift'])
    print('grid: ', grid.GetNumberOfPoints(), grid.GetNumberOfCells(), zShift.shape)
    writeVTK(grid, '/Users/dpn/quad.vtk')

    return (grid, nx, ny)

def duplicate3DMesh(grid, numDup) :
    gridDup = vtk.vtkUnstructuredGrid()
    ptsDup = vtk.vtkPoints()
    gridDup.SetPoints(ptsDup)

    pts = grid.GetPoints()
    phi = 0.0
    dPhi = 2*math.pi / numDup
    numPts = grid.GetNumberOfPoints()
    numCells = grid.GetNumberOfCells()

    dupI = vtk.vtkFloatArray()
    dupJ = vtk.vtkFloatArray()
    dupK = vtk.vtkFloatArray()
    dupI.SetName('I')
    dupJ.SetName('J')
    dupK.SetName('K')
    gridDup.GetCellData().AddArray(dupI)
    gridDup.GetCellData().AddArray(dupJ)
    gridDup.GetCellData().AddArray(dupK)


    ptOffset = 0
    varI = grid.GetCellData().GetArray('I')
    varJ = grid.GetCellData().GetArray('J')
    varK = grid.GetCellData().GetArray('K')
    _numDup = numDup
    _numDup = 5
    for i in range(_numDup) :
        for j in range(numPts) :
            _pt = pts.GetPoint(j)
            pt = (_pt[0], _pt[1], _pt[2]+phi)
            ptsDup.InsertNextPoint(pt)
        for j in range(numCells) :
            cell = grid.GetCell(j)
            ptIds = cell.GetPointIds()
            n = ptIds.GetNumberOfIds()
            dupIds = []
            for k in range(n) :
                dupIds.append(ptIds.GetId(k) + ptOffset)
            gridDup.InsertNextCell(vtk.VTK_HEXAHEDRON, 8, dupIds)

            dupI.InsertNextValue(varI.GetValue(j))
            dupJ.InsertNextValue(varJ.GetValue(j))
            dupK.InsertNextValue(varK.GetValue(j))

        phi = phi + dPhi
        ptOffset = ptOffset + numPts

    writeVTK(gridDup, '/Users/dpn/dupGrid.vtk')


##MAIN

# Open the NetCDF file
ncgrid = xr.open_dataset('./1_5_torus/65402_68x32_revIp_wide_fixBp_curv.nc')
data = ad.FileReader('./1_5_torus/BOUT.dmp.bp')

guardJ = 2
(grid,nx,ny) = buildQuadMesh(ncgrid, data, guardJ)
grid = addZShiftStuff(grid, data, nx,ny, guardJ)
grid = build3DMesh(grid, data, nx, ny, guardJ)

numDup = 5
grid = duplicate3DMesh(grid, numDup)


meow()

zShift = ds['zShift'].values
phi = ds['phi'].values
grid = vtk.vtkImageData()
grid.SetDimensions(68+1,64+1,1)
var = vtk.vtkFloatArray()
phiV = vtk.vtkFloatArray()
var.SetName('zShift')
phiV.SetName('phi')
grid.GetCellData().AddArray(var)
grid.GetCellData().AddArray(phiV)
cnt = 0
for j in range(64) :
    for i in range(68) :
        #var.InsertNextValue(cnt)
        var.InsertNextValue(zShift[i,j])
        phiV.InsertNextValue(phi[i,j,0])
        cnt = cnt+1
writeVTK(grid, '/Users/dpn/uniform.vtk')

grid = buildQuadMesh(ds['phi'])
grid = addZShiftStuff(grid, ds)
build3DMesh(grid, ds)
print('\n\n********\n All done.\n\n')
meow()

lowerLeft = (ds['Rxy_lower_left_corners'], ds['Zxy_lower_left_corners'])
upperLeft = (ds['Rxy_upper_left_corners'], ds['Zxy_upper_left_corners'])
lowerRight = (ds['Rxy_lower_right_corners'], ds['Zxy_lower_right_corners'])
upperRight = (ds['Rxy_upper_right_corners'], ds['Zxy_upper_right_corners'])
dumpGrid(lowerLeft, upperLeft, lowerRight, upperRight, None)

dataShifted = doFFT(ds, 'phi')
print(' done with fft...')

