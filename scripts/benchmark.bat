@echo off
setlocal enabledelayedexpansion

:: Usage: benchmark.bat <configs_folder>
set CONFIGS_DIR=%~1
if "%CONFIGS_DIR%"=="" (
    echo Usage: benchmark.bat ^<configs_folder^>
    exit /b 1
)

set FOUND=0
for %%f in ("%CONFIGS_DIR%\*.toml") do (
    set FOUND=1
    echo.
    echo ========================================
    echo Config: %%~nxf
    echo ========================================
    call "%~dp0run_five_times.bat" "%%f"
    if errorlevel 1 (
        echo ERROR: benchmark failed for %%~nxf
        exit /b 1
    )
)

if !FOUND!==0 (
    echo No .toml files found in %CONFIGS_DIR%
    exit /b 1
)

echo.
echo ========================================
echo Benchmark complete.
echo ========================================
