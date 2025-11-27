# LichtFeld-Studio One-Shot Build Script for Windows
# This script verifies prerequisites, sets up dependencies, and builds the project
# Usage: .\build_lichtfeld.ps1 [-Configuration Debug|Release] [-Clean] [-Help]

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Release',

    [switch]$SkipVerification,
    [switch]$SkipVcpkg,
    [switch]$SkipLibTorch,
    [switch]$Clean,
    [switch]$Help
)

if ($Help) {
    Write-Host @"
LichtFeld-Studio One-Shot Build Script

Usage: .\build_lichtfeld.ps1 [options]

This script automatically:
  1. Verifies build prerequisites (VS 2022, CUDA 12.8, CMake, Git)
  2. Sets up vcpkg in the parent directory
  3. Downloads LibTorch (Debug & Release) if missing
  4. Configures and builds LichtFeld-Studio

Options:
  -Configuration <Debug|Release>  Build configuration (default: Release)
  -SkipVerification               Skip environment verification
  -SkipVcpkg                      Skip vcpkg setup
  -SkipLibTorch                   Skip LibTorch download
  -Clean                          Clean build directory before building
  -Help                           Show this help message

Examples:
  .\build_lichtfeld.ps1                        Build Release (default)
  .\build_lichtfeld.ps1 -Configuration Debug   Build Debug
  .\build_lichtfeld.ps1 -Clean                 Clean and rebuild
  .\build_lichtfeld.ps1 -SkipLibTorch          Skip LibTorch download (if already present)
"@
    exit 0
}

$ErrorActionPreference = 'Stop'
$ScriptPath = $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptPath
$VcpkgPath = Join-Path (Split-Path -Parent $ProjectRoot) "vcpkg"

# Track overall status
$AllChecksPassed = $true
$FailedChecks = @()

# ============================================================================
# Helper Functions
# ============================================================================

function Test-Command {
    param([string]$Command)
    try {
        if (Get-Command $Command -ErrorAction SilentlyContinue) {
            return $true
        }
    } catch {
        return $false
    }
    return $false
}

function Write-Status {
    param(
        [string]$Name,
        [bool]$Passed,
        [string]$Version = "",
        [string]$ErrorMessage = "",
        [string]$Solution = ""
    )

    $StatusSymbol = if ($Passed) { "[OK]" } else { "[FAIL]" }
    $StatusColor = if ($Passed) { "Green" } else { "Red" }

    Write-Host ("{0,-35}" -f $Name) -NoNewline
    Write-Host $StatusSymbol -ForegroundColor $StatusColor -NoNewline

    if ($Version -ne "") {
        Write-Host " - $Version" -ForegroundColor Gray
    } else {
        Write-Host ""
    }

    if (-not $Passed) {
        if ($ErrorMessage -ne "") {
            Write-Host "  ERROR: $ErrorMessage" -ForegroundColor Red
        }
        if ($Solution -ne "") {
            Write-Host "  SOLUTION: $Solution" -ForegroundColor Yellow
        }
        Write-Host ""
        $script:AllChecksPassed = $false
        $script:FailedChecks += $Name
    }
}

function Write-Warning-Status {
    param(
        [string]$Name,
        [string]$Message,
        [string]$Suggestion = ""
    )
    Write-Host ("{0,-35}" -f $Name) -NoNewline
    Write-Host "[WARN]" -ForegroundColor Yellow -NoNewline
    Write-Host " - $Message" -ForegroundColor Gray
    if ($Suggestion -ne "") {
        Write-Host "  SUGGESTION: $Suggestion" -ForegroundColor Cyan
    }
}

# ============================================================================
# Auto-Launch VS Developer Environment
# ============================================================================

function Test-VSDevEnvironment {
    return ($env:VSINSTALLDIR -or $env:VisualStudioVersion)
}

function Find-VSInstallPath {
    $VSWherePath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

    if (-not (Test-Path $VSWherePath)) {
        return $null
    }

    try {
        # Try full VS IDE first
        $VSPath = & $VSWherePath -latest -products Microsoft.VisualStudio.Product.Community,Microsoft.VisualStudio.Product.Professional,Microsoft.VisualStudio.Product.Enterprise -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null

        if (-not $VSPath) {
            # Try Build Tools
            $VSPath = & $VSWherePath -latest -products Microsoft.VisualStudio.Product.BuildTools -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        }

        return $VSPath
    } catch {
        return $null
    }
}

function Launch-VSDevEnvironment {
    $VSInstallPath = Find-VSInstallPath

    if (-not $VSInstallPath) {
        Write-Host "ERROR: Visual Studio 2022 with C++ workload not found!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please install Visual Studio 2022 with 'Desktop development with C++' workload" -ForegroundColor Yellow
        Write-Host "Download from: https://visualstudio.microsoft.com/downloads/" -ForegroundColor Cyan
        exit 1
    }

    $VSDevShellScript = Join-Path $VSInstallPath "Common7\Tools\Launch-VsDevShell.ps1"

    if (-not (Test-Path $VSDevShellScript)) {
        Write-Host "ERROR: VS Developer PowerShell module not found at expected location" -ForegroundColor Red
        exit 1
    }

    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "Launching VS Developer Environment" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Not running in VS Developer Command Prompt." -ForegroundColor Yellow
    Write-Host "Re-launching script in VS Developer PowerShell..." -ForegroundColor Yellow
    Write-Host ""

    # Build parameter string for re-launch
    $ParamString = ""
    if ($Configuration -ne 'Release') { $ParamString += " -Configuration $Configuration" }
    if ($SkipVerification) { $ParamString += " -SkipVerification" }
    if ($SkipVcpkg) { $ParamString += " -SkipVcpkg" }
    if ($SkipLibTorch) { $ParamString += " -SkipLibTorch" }
    if ($Clean) { $ParamString += " -Clean" }

    # Create temp script to launch dev shell and run build
    $TempScript = Join-Path $env:TEMP "lichtfeld_build_$(Get-Random).ps1"

    @"
# Initialize VS Developer Environment
& '$VSDevShellScript' -Arch amd64 -HostArch amd64

# Change to project directory
Set-Location '$ProjectRoot'

Write-Host ''
Write-Host '================================================================' -ForegroundColor Green
Write-Host 'VS Developer Environment Initialized!' -ForegroundColor Green
Write-Host '================================================================' -ForegroundColor Green
Write-Host ''

# Run the build script with parameters
& '$ScriptPath'$ParamString

Write-Host ''
Write-Host 'Press any key to exit...' -ForegroundColor Cyan
`$null = `$Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
"@ | Out-File -FilePath $TempScript -Encoding UTF8

    # Launch new PowerShell window
    Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -NoExit -File `"$TempScript`""

    Write-Host "Build script launched in a new window." -ForegroundColor Green
    Write-Host "This window will now close." -ForegroundColor Gray
    Start-Sleep -Seconds 2
    exit 0
}

# Check if we need to launch VS Dev environment
if (-not (Test-VSDevEnvironment)) {
    Launch-VSDevEnvironment
}

# ============================================================================
# Environment Verification
# ============================================================================

function Test-BuildEnvironment {
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "Verifying Build Environment" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""

    # Check 1: PowerShell Version
    Write-Host "[1/7] Checking PowerShell..." -ForegroundColor Yellow
    $PSVer = $PSVersionTable.PSVersion
    if ($PSVer.Major -ge 5) {
        Write-Status "PowerShell" $true "v$($PSVer.Major).$($PSVer.Minor)"
    } else {
        Write-Status "PowerShell" $false "v$($PSVer.Major).$($PSVer.Minor)" `
            "PowerShell 5.0 or later required" `
            "Update PowerShell: https://docs.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-windows"
    }

    # Check 2: Visual Studio / MSVC
    Write-Host "[2/7] Checking Visual Studio..." -ForegroundColor Yellow
    if (Test-Command "cl") {
        try {
            $ClVersion = & cl 2>&1 | Select-String "Version" | Select-Object -First 1
            Write-Status "MSVC Compiler (cl.exe)" $true $ClVersion.ToString().Trim()
        } catch {
            Write-Status "MSVC Compiler (cl.exe)" $true "Found"
        }
    } else {
        Write-Status "MSVC Compiler (cl.exe)" $false "" `
            "Visual Studio Build Tools not found" `
            "Install Visual Studio 2022 with 'Desktop development with C++'"
    }

    # Check 3: CMake
    Write-Host "[3/7] Checking CMake..." -ForegroundColor Yellow
    if (Test-Command "cmake") {
        try {
            $CMakeVersion = (cmake --version | Select-Object -First 1) -replace 'cmake version ', ''
            $CMakeVersionParsed = [version]($CMakeVersion -split '-')[0]

            if ($CMakeVersionParsed -ge [version]"3.30") {
                Write-Status "CMake" $true "v$CMakeVersion"
            } else {
                Write-Status "CMake" $false "v$CMakeVersion (too old)" `
                    "CMake 3.30 or later is required" `
                    "Download from: https://cmake.org/download/"
            }
        } catch {
            Write-Status "CMake" $true "Found (version check failed)"
        }
    } else {
        Write-Status "CMake" $false "" `
            "CMake not found in PATH" `
            "Download CMake 3.30+ from: https://cmake.org/download/"
    }

    # Check 4: CUDA Toolkit
    Write-Host "[4/7] Checking CUDA Toolkit 12.8..." -ForegroundColor Yellow
    if (Test-Command "nvcc") {
        try {
            $NvccOutput = nvcc --version 2>&1 | Select-String "release"
            $CudaVersion = ($NvccOutput -split "release ")[1] -split "," | Select-Object -First 1

            if ($CudaVersion -match "12\.8") {
                Write-Status "CUDA Toolkit (nvcc)" $true "v$CudaVersion"
            } else {
                Write-Status "CUDA Toolkit (nvcc)" $false "v$CudaVersion" `
                    "CUDA 12.8 is required (found $CudaVersion)" `
                    "Download CUDA 12.8 from: https://developer.nvidia.com/cuda-12-8-0-download-archive"
            }
        } catch {
            Write-Status "CUDA Toolkit (nvcc)" $true "Found (version check failed)"
        }
    } else {
        Write-Status "CUDA Toolkit (nvcc)" $false "" `
            "CUDA Toolkit not found or nvcc not in PATH" `
            "Download CUDA 12.8 from: https://developer.nvidia.com/cuda-12-8-0-download-archive"
    }

    # Check 5: Git
    Write-Host "[5/7] Checking Git..." -ForegroundColor Yellow
    if (Test-Command "git") {
        try {
            $GitVersion = (git --version) -replace 'git version ', ''
            Write-Status "Git" $true "v$GitVersion"
        } catch {
            Write-Status "Git" $true "Found"
        }
    } else {
        Write-Status "Git" $false "" `
            "Git not found in PATH" `
            "Download Git from: https://git-scm.com/download/win"
    }

    # Check 6: Ninja (optional but recommended)
    Write-Host "[6/7] Checking Ninja (optional)..." -ForegroundColor Yellow
    if (Test-Command "ninja") {
        try {
            $NinjaVersion = ninja --version
            Write-Status "Ninja" $true "v$NinjaVersion"
        } catch {
            Write-Status "Ninja" $true "Found"
        }
    } else {
        Write-Warning-Status "Ninja" "Not found (build will use fallback generator)" `
            "Install via: choco install ninja OR download from https://github.com/ninja-build/ninja/releases"
    }

    # Check 7: Disk Space
    Write-Host "[7/7] Checking disk space..." -ForegroundColor Yellow
    try {
        $Drive = (Get-Location).Drive
        $FreeSpaceGB = [math]::Round((Get-PSDrive $Drive.Name).Free / 1GB, 2)

        if ($FreeSpaceGB -gt 30) {
            Write-Status "Disk Space" $true "$FreeSpaceGB GB available"
        } elseif ($FreeSpaceGB -gt 15) {
            Write-Warning-Status "Disk Space" "$FreeSpaceGB GB available (may be insufficient)" `
                "Build may require 20-30 GB. Consider freeing up space."
        } else {
            Write-Status "Disk Space" $false "$FreeSpaceGB GB available" `
                "Insufficient disk space (at least 15-20 GB recommended)" `
                "Free up disk space on drive $($Drive.Name):"
        }
    } catch {
        Write-Warning-Status "Disk Space" "Could not determine free space"
    }

    Write-Host ""

    return $script:AllChecksPassed
}

# ============================================================================
# vcpkg Setup
# ============================================================================

function Setup-Vcpkg {
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "Setting Up vcpkg" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "vcpkg location: $VcpkgPath" -ForegroundColor Gray
    Write-Host ""

    if (-not (Test-Path $VcpkgPath)) {
        Write-Host "Cloning vcpkg repository..." -ForegroundColor Yellow
        git clone https://github.com/microsoft/vcpkg.git "$VcpkgPath"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to clone vcpkg" -ForegroundColor Red
            exit 1
        }
        Write-Host "vcpkg cloned successfully!" -ForegroundColor Green
    } else {
        Write-Host "vcpkg directory exists. Pulling latest changes..." -ForegroundColor Yellow
        Push-Location $VcpkgPath
        try {
            git pull
            if ($LASTEXITCODE -ne 0) {
                Write-Host "WARNING: git pull failed, continuing with existing version" -ForegroundColor Yellow
            } else {
                Write-Host "vcpkg updated successfully!" -ForegroundColor Green
            }
        } finally {
            Pop-Location
        }
    }

    # Bootstrap vcpkg if needed
    $VcpkgExe = Join-Path $VcpkgPath "vcpkg.exe"
    if (-not (Test-Path $VcpkgExe)) {
        Write-Host ""
        Write-Host "Bootstrapping vcpkg..." -ForegroundColor Yellow
        Write-Host "This may take a few minutes..." -ForegroundColor Gray

        Push-Location $VcpkgPath
        try {
            & .\bootstrap-vcpkg.bat -disableMetrics
            if ($LASTEXITCODE -ne 0) {
                Write-Host "ERROR: Failed to bootstrap vcpkg" -ForegroundColor Red
                exit 1
            }
        } finally {
            Pop-Location
        }
        Write-Host "vcpkg bootstrapped successfully!" -ForegroundColor Green
    } else {
        Write-Host "vcpkg.exe already exists. Skipping bootstrap." -ForegroundColor Green
    }

    # Set environment variable for CMake
    $env:VCPKG_ROOT = $VcpkgPath
    Write-Host ""
    Write-Host "VCPKG_ROOT set to: $VcpkgPath" -ForegroundColor Cyan
    Write-Host ""
}

# ============================================================================
# LibTorch Download
# ============================================================================

function Setup-LibTorch {
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "Setting Up LibTorch" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""

    $ExternalDir = Join-Path $ProjectRoot "external"
    $DebugDir = Join-Path $ExternalDir "debug"
    $ReleaseDir = Join-Path $ExternalDir "release"
    $DebugLibTorch = Join-Path $DebugDir "libtorch"
    $ReleaseLibTorch = Join-Path $ReleaseDir "libtorch"

    # Create directories
    if (-not (Test-Path $ExternalDir)) {
        New-Item -ItemType Directory -Path $ExternalDir | Out-Null
        Write-Host "Created: external/" -ForegroundColor Gray
    }
    if (-not (Test-Path $DebugDir)) {
        New-Item -ItemType Directory -Path $DebugDir | Out-Null
        Write-Host "Created: external/debug/" -ForegroundColor Gray
    }
    if (-not (Test-Path $ReleaseDir)) {
        New-Item -ItemType Directory -Path $ReleaseDir | Out-Null
        Write-Host "Created: external/release/" -ForegroundColor Gray
    }

    # Download Debug LibTorch if missing
    if (-not (Test-Path $DebugLibTorch)) {
        Write-Host ""
        Write-Host "Downloading LibTorch (Debug)..." -ForegroundColor Yellow
        Write-Host "This is a large download (~3.2 GB). Please wait..." -ForegroundColor Gray

        $DebugZip = Join-Path $ProjectRoot "libtorch-debug.zip"
        $DebugUrl = "https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-debug-2.7.0%2Bcu128.zip"

        try {
            curl.exe -L -o "$DebugZip" "$DebugUrl" --progress-bar
            if ($LASTEXITCODE -ne 0) { throw "Download failed" }

            Write-Host "Extracting LibTorch (Debug)..." -ForegroundColor Yellow
            tar -xf "$DebugZip" -C "$DebugDir"
            if ($LASTEXITCODE -ne 0) { throw "Extraction failed" }

            Remove-Item $DebugZip -Force
            Write-Host "LibTorch (Debug) installed successfully!" -ForegroundColor Green
        } catch {
            Write-Host "ERROR: Failed to download/extract Debug LibTorch: $_" -ForegroundColor Red
            if (Test-Path $DebugZip) { Remove-Item $DebugZip -Force }
            exit 1
        }
    } else {
        Write-Host "LibTorch (Debug) already exists. Skipping download." -ForegroundColor Green
    }

    # Download Release LibTorch if missing
    if (-not (Test-Path $ReleaseLibTorch)) {
        Write-Host ""
        Write-Host "Downloading LibTorch (Release)..." -ForegroundColor Yellow
        Write-Host "This is a large download (~3.2 GB). Please wait..." -ForegroundColor Gray

        $ReleaseZip = Join-Path $ProjectRoot "libtorch-release.zip"
        $ReleaseUrl = "https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-2.7.0%2Bcu128.zip"

        try {
            curl.exe -L -o "$ReleaseZip" "$ReleaseUrl" --progress-bar
            if ($LASTEXITCODE -ne 0) { throw "Download failed" }

            Write-Host "Extracting LibTorch (Release)..." -ForegroundColor Yellow
            tar -xf "$ReleaseZip" -C "$ReleaseDir"
            if ($LASTEXITCODE -ne 0) { throw "Extraction failed" }

            Remove-Item $ReleaseZip -Force
            Write-Host "LibTorch (Release) installed successfully!" -ForegroundColor Green
        } catch {
            Write-Host "ERROR: Failed to download/extract Release LibTorch: $_" -ForegroundColor Red
            if (Test-Path $ReleaseZip) { Remove-Item $ReleaseZip -Force }
            exit 1
        }
    } else {
        Write-Host "LibTorch (Release) already exists. Skipping download." -ForegroundColor Green
    }

    Write-Host ""
}

# ============================================================================
# Build LichtFeld-Studio
# ============================================================================

function Build-LichtFeldStudio {
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "Building LichtFeld-Studio ($Configuration)" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""

    Push-Location $ProjectRoot
    try {
        $BuildDir = Join-Path $ProjectRoot "build"
        $VcpkgToolchain = Join-Path $VcpkgPath "scripts\buildsystems\vcpkg.cmake"

        # Verify vcpkg toolchain exists
        if (-not (Test-Path $VcpkgToolchain)) {
            Write-Host "ERROR: vcpkg toolchain file not found!" -ForegroundColor Red
            Write-Host "Expected: $VcpkgToolchain" -ForegroundColor Gray
            Write-Host ""
            Write-Host "Please run without -SkipVcpkg to set up vcpkg first." -ForegroundColor Yellow
            exit 1
        }

        # Verify LibTorch exists for the selected configuration
        $LibTorchPath = if ($Configuration -eq 'Debug') {
            Join-Path $ProjectRoot "external\debug\libtorch"
        } else {
            Join-Path $ProjectRoot "external\release\libtorch"
        }

        if (-not (Test-Path $LibTorchPath)) {
            Write-Host "ERROR: LibTorch ($Configuration) not found!" -ForegroundColor Red
            Write-Host "Expected: $LibTorchPath" -ForegroundColor Gray
            Write-Host ""
            Write-Host "Please run without -SkipLibTorch to download LibTorch first." -ForegroundColor Yellow
            exit 1
        }

        # Clean if requested
        if ($Clean -and (Test-Path $BuildDir)) {
            Write-Host "Cleaning build directory..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force $BuildDir
            Write-Host "Build directory cleaned." -ForegroundColor Green
            Write-Host ""
        }

        # Determine generator
        $Generator = "Ninja"
        if (-not (Test-Command "ninja")) {
            Write-Host "Ninja not found. Using Visual Studio generator..." -ForegroundColor Yellow
            $Generator = "Visual Studio 17 2022"
        }

        # Configure
        Write-Host "Configuring CMake..." -ForegroundColor Yellow
        Write-Host "  Generator: $Generator" -ForegroundColor Gray
        Write-Host "  Configuration: $Configuration" -ForegroundColor Gray
        Write-Host "  Toolchain: $VcpkgToolchain" -ForegroundColor Gray
        Write-Host ""

        if ($Generator -eq "Ninja") {
            cmake -B build `
                "-DCMAKE_BUILD_TYPE=$Configuration" `
                -G "$Generator" `
                "-DCMAKE_TOOLCHAIN_FILE=$VcpkgToolchain"
        } else {
            # VS generator doesn't use CMAKE_BUILD_TYPE at configure time
            cmake -B build `
                -G "$Generator" `
                -A x64 `
                "-DCMAKE_TOOLCHAIN_FILE=$VcpkgToolchain"
        }

        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: CMake configuration failed!" -ForegroundColor Red
            exit 1
        }

        Write-Host ""
        Write-Host "CMake configuration successful!" -ForegroundColor Green
        Write-Host ""

        # Build
        Write-Host "Building LichtFeld-Studio..." -ForegroundColor Yellow
        Write-Host "This may take 10-30 minutes depending on your system..." -ForegroundColor Gray
        Write-Host ""

        if ($Generator -eq "Ninja") {
            cmake --build build -j
        } else {
            cmake --build build --config "$Configuration" -j
        }

        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Build failed!" -ForegroundColor Red
            exit 1
        }

        Write-Host ""
        Write-Host "================================================================" -ForegroundColor Green
        Write-Host "Build Successful!" -ForegroundColor Green
        Write-Host "================================================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Executable location:" -ForegroundColor Cyan

        if ($Generator -eq "Ninja") {
            Write-Host "  $BuildDir\LichtFeld-Studio.exe" -ForegroundColor White
        } else {
            Write-Host "  $BuildDir\$Configuration\LichtFeld-Studio.exe" -ForegroundColor White
        }
        Write-Host ""

    } finally {
        Pop-Location
    }
}

# ============================================================================
# Main Execution
# ============================================================================

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "LichtFeld-Studio One-Shot Build Script" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project: $ProjectRoot" -ForegroundColor Gray
Write-Host "Configuration: $Configuration" -ForegroundColor Gray
Write-Host ""

# Phase 1: Environment Verification
if (-not $SkipVerification) {
    $EnvOK = Test-BuildEnvironment
    if (-not $EnvOK) {
        Write-Host "================================================================" -ForegroundColor Red
        Write-Host "Environment Verification Failed" -ForegroundColor Red
        Write-Host "================================================================" -ForegroundColor Red
        Write-Host ""
        Write-Host "Failed checks:" -ForegroundColor Red
        foreach ($Check in $FailedChecks) {
            Write-Host "  - $Check" -ForegroundColor Red
        }
        Write-Host ""
        Write-Host "Please fix the issues above and run this script again." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "Environment Verification Passed!" -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "Skipping environment verification (-SkipVerification)" -ForegroundColor Yellow
    Write-Host ""
}

# Phase 2: vcpkg Setup
if (-not $SkipVcpkg) {
    Setup-Vcpkg
} else {
    Write-Host "Skipping vcpkg setup (-SkipVcpkg)" -ForegroundColor Yellow
    # Still set VCPKG_ROOT if vcpkg exists
    if (Test-Path $VcpkgPath) {
        $env:VCPKG_ROOT = $VcpkgPath
    } else {
        Write-Host "WARNING: vcpkg not found at $VcpkgPath" -ForegroundColor Yellow
        Write-Host "Build may fail. Consider running without -SkipVcpkg" -ForegroundColor Yellow
    }
    Write-Host ""
}

# Phase 3: LibTorch Download
if (-not $SkipLibTorch) {
    Setup-LibTorch
} else {
    Write-Host "Skipping LibTorch setup (-SkipLibTorch)" -ForegroundColor Yellow
    Write-Host ""
}

# Phase 4: Build
Build-LichtFeldStudio

Write-Host ""
Write-Host "All done!" -ForegroundColor Green
Write-Host ""
