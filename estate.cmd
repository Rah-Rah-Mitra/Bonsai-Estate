@echo off
rem Sample Town N5 estate pipeline - runs on Blender 5.2's bundled Python (Bonsai / IfcOpenShell 0.9).
"C:\Program Files\Blender Foundation\Blender 5.2\5.2\python\bin\python.exe" -I -B "%~dp0estate.py" %*
