@echo off
setlocal enabledelayedexpansion

:: Usage: run_five_times.bat <config_file>
set CONFIG_FILE=%~1
if "%CONFIG_FILE%"=="" (
    echo Usage: run_five_times.bat ^<config_file^>
    exit /b 1
)

:: Read output_dir from the TOML via Python (falls back to ./output/default_output)
for /f "usebackq delims=" %%i in (`python -c "from marl_path.shared.config import load_config; c=load_config(r'%CONFIG_FILE%'); print(c.get('output_dir','./output/default_output'))"`) do set BASE_OUTPUT=%%i

echo Config  : %CONFIG_FILE%
echo Base out: %BASE_OUTPUT%
echo.

for /l %%n in (1,1,5) do (
    echo [run %%n/5] seed=%%n  output=%BASE_OUTPUT%\run%%n
    marl-path --config-file "%CONFIG_FILE%" --seed %%n --seed-training %%n --output-dir "%BASE_OUTPUT%\run%%n"
    if errorlevel 1 (
        echo ERROR: run %%n failed for %CONFIG_FILE%
        exit /b 1
    )
)

echo.
echo All 5 runs finished for %CONFIG_FILE%
