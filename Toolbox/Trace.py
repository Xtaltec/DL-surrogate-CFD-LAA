# trace generated using paraview version 5.6.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'EnSight Reader'

lAA_1encas = EnSightReader(CaseFileName=EncasDir)

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1563, 806]

# show data in view
lAA_1encasDisplay = Show(lAA_1encas, renderView1)

# trace defaults for the display properties.
lAA_1encasDisplay.Representation = 'Surface'

# reset view to fit data
renderView1.ResetCamera()

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
lAA_1encasDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get color transfer function/color map for 'z_wall_shear'
z_wall_shearLUT = GetColorTransferFunction('z_wall_shear')

# get opacity transfer function/opacity map for 'z_wall_shear'
z_wall_shearPWF = GetOpacityTransferFunction('z_wall_shear')

# save data
SaveData('D:/Simulations/EncaseFiles/LAA_1/ECAP/LAA.csv', proxy=lAA_1encas, WriteTimeSteps=1)

# create a new 'Extract Surface'
extractSurface1 = ExtractSurface(Input=lAA_1encas)

# show data in view
extractSurface1Display = Show(extractSurface1, renderView1)

# trace defaults for the display properties.
extractSurface1Display.Representation = 'Surface'

# hide data in view
Hide(lAA_1encas, renderView1)

# show color bar/color legend
extractSurface1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# save data
SaveData('D:/Simulations/EncaseFiles/LAA_1/ECAP/LAA.stl', proxy=extractSurface1, FileType='Ascii')

# hide data in view
Hide(extractSurface1, renderView1)

# create a new 'STL Reader'
EncasDir='D:\\Simulations\\EncaseFiles\\LAA_1\\ECAP\\LAA_11.stl
lAA1stl = STLReader(FileNames=[EncasDir])

# show data in view
lAA1stlDisplay = Show(lAA1stl, renderView1)

# trace defaults for the display properties.
lAA1stlDisplay.Representation = 'Surface'

# reset view to fit data
renderView1.ResetCamera()

# show color bar/color legend
lAA1stlDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get color transfer function/color map for 'STLSolidLabeling'
sTLSolidLabelingLUT = GetColorTransferFunction('STLSolidLabeling')

# get opacity transfer function/opacity map for 'STLSolidLabeling'
sTLSolidLabelingPWF = GetOpacityTransferFunction('STLSolidLabeling')

# save data
SaveData('D:\Simulations\EncaseFiles\LAA_1\ECAP\LAA_SURFACE.vtk', proxy=lAA1stl, FileType='Ascii')