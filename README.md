MayaAWD
=======

AWD exporter for Maya X64 win
This is just a copy of files needed to get AWD export in Maya x64 win.
The originals were posted on the Away3D forum, im just sticking them here.

Put the files, folders in your MAYA install Folder
eg
C:\Program Files\Autodesk\Maya2013

The exporter works with animation (bones) and meshes.

Note.
This exporter does not merge subgeometries. So meshes with more than 1900 tris will fail.
You could just split your mesh export to AWD and use something like Prefab to merge to a single mesh.

Export Selection doesnt work.

Use export all and hide any meshes you dont want exported.
