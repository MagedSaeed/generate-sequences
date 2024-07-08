@REM @ECHO OFF

@REM pushd %~dp0

@REM REM Command file for Sphinx documentation

@REM if "%SPHINXBUILD%" == "" (
@REM 	set SPHINXBUILD=sphinx-build
@REM )
@REM set SOURCEDIR=source
@REM set BUILDDIR=build

@REM if "%1" == "" goto help

@REM %SPHINXBUILD% >NUL 2>NUL
@REM if errorlevel 9009 (
@REM 	echo.
@REM 	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
@REM 	echo.installed, then set the SPHINXBUILD environment variable to point
@REM 	echo.to the full path of the 'sphinx-build' executable. Alternatively you
@REM 	echo.may add the Sphinx directory to PATH.
@REM 	echo.
@REM 	echo.If you don't have Sphinx installed, grab it from
@REM 	echo.https://www.sphinx-doc.org/
@REM 	exit /b 1
@REM )

@REM %SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
@REM goto end

@REM :help
@REM %SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

@REM :end
@REM popd
