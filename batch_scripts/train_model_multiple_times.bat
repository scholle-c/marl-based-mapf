@echo off
setlocal enabledelayedexpansion

REM Usage: train_model_multiple_times.bat <config_path> <num_runs> [base_output_dir]
REM Example: train_model_multiple_times.bat configs/examples/example_train_from_data.json 5 output

if "%~1"=="" (
    echo Usage: %~nx0 ^<config_path^> ^<num_runs^> [base_output_dir]
    exit /b 1
)

set "CONFIG=%~1"
set "NUM_RUNS=%~2"
if "%NUM_RUNS%"=="" set "NUM_RUNS=1"

set "BASE_OUTPUT=%~3"
if "%BASE_OUTPUT%"=="" set "BASE_OUTPUT=output"

echo Running %NUM_RUNS% training runs using config "%CONFIG%"

for /l %%i in (1,1,%NUM_RUNS%) do (
    set "SEED=%%i"
    set "RUN_DIR=%BASE_OUTPUT%\run_%%i"
    echo.
    echo ==== Run %%i of %NUM_RUNS% ^| seed=!SEED! ^| output=!RUN_DIR! ====
    python app.py train --config "%CONFIG%" --seed !SEED! --output-dir "!RUN_DIR!"
    if errorlevel 1 (
        echo Training run %%i failed. Aborting.
        exit /b 1
    )
)

echo.
echo ==== Plotting aggregated loss curve ====
set "HISTORY_ARGS="
for /l %%i in (1,1,%NUM_RUNS%) do (
    set "HISTORY_PATH=%BASE_OUTPUT%\run_%%i\loss_history.json"
    if exist "!HISTORY_PATH!" (
        set "HISTORY_ARGS=!HISTORY_ARGS! !HISTORY_PATH!"
    )
)

if "%HISTORY_ARGS%"=="" (
    echo No loss_history.json files found under %BASE_OUTPUT%\run_*; skipping plot.
) else (
    python app.py eval --mode loss --history-path !HISTORY_ARGS! --loss-output "%BASE_OUTPUT%\loss_curve.png"
    if errorlevel 1 (
        echo Plotting failed.
        exit /b 1
    )
)

echo.
echo All runs completed.
endlocal
