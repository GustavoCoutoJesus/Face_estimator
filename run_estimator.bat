@echo off
setlocal
cd /d "%~dp0"

echo ============================================================================
echo   Social Judgment Face Estimator (Todorov ^& Peterson OMI - 34 Attributes)
echo ============================================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not on your PATH.
    echo Please install Python 3.10 or 3.11 from https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist "input_faces" mkdir "input_faces"

python -c "import face_trait_transformer, mediapipe, cv2, torch" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [1/2] Installing required Python packages from requirements.txt...
    python -m pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Package installation failed. Check the messages above.
        pause
        exit /b 1
    )
)

echo Running face estimation on images in .\input_faces ...
echo.
python estimate_faces.py input_faces --output-dir results

echo.
echo ============================================================================
echo Done! Check the .\results folder for:
echo   - social_judgments.csv
echo   - figures\                 (Social traits + Todorov 2D space)
echo   - figures_demographics\    (Physical, demographic ^& ethnic attributions)
echo   - aligned_faces\           (FFHQ-aligned 512x512 crops)
echo ============================================================================
pause
