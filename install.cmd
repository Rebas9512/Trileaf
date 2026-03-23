@echo off
setlocal

rem Trileaf Windows bootstrap for cmd.exe
rem Usage:
rem   curl -fsSL https://raw.githubusercontent.com/Rebas9512/Trileaf/main/install.cmd -o install.cmd && install.cmd && del install.cmd

set "SCRIPT_URL=%TRILEAF_INSTALL_PS1_URL%"
if not defined SCRIPT_URL set "SCRIPT_URL=https://raw.githubusercontent.com/Rebas9512/Trileaf/main/install.ps1"
set "SCRIPT_PATH=%TEMP%\trileaf-install-%RANDOM%%RANDOM%.ps1"

powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -UseBasicParsing '%SCRIPT_URL%' -OutFile '%SCRIPT_PATH%' } catch { Write-Host $_; exit 1 }"
if errorlevel 1 (
    echo Failed to download %SCRIPT_URL%
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_PATH%" %*
set "EXITCODE=%ERRORLEVEL%"

del "%SCRIPT_PATH%" >nul 2>&1

rem Refresh PATH in this CMD session so the new binary is usable immediately.
if %EXITCODE% equ 0 (
    endlocal
    for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "PATH=%%b;%PATH%"
) else (
    endlocal
)
exit /b %EXITCODE%
