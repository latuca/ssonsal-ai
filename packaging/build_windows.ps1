<#
.SYNOPSIS
  임베디드 Python 기반 Windows 설치본을 만든다. (packaging/README.md 참고)
#>
param(
    [string]$PythonVersion = "3.10.11",   # Windows용 임베디드 배포가 있는 마지막 3.10
    [string]$FfmpegExe = "",              # 비우면 gyan.dev essentials 빌드를 내려받는다
    [string]$ModelFile = "",              # 넣으면 models/에 함께 담는다(v2)
    [switch]$SkipInstaller
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Build = Join-Path $PSScriptRoot "build"
$App = Join-Path $Build "Acappella"
$Py = Join-Path $App "python"
$Dist = Join-Path $PSScriptRoot "dist"
$Cache = Join-Path $Build "cache"

Remove-Item -Recurse -Force $App -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $Py, $Cache, $Dist, (Join-Path $App "bin"), (Join-Path $App "models") | Out-Null

function Get-Cached($Url, $Name) {
    $dst = Join-Path $Cache $Name
    if (-not (Test-Path $dst)) {
        Write-Host "내려받는 중: $Url"
        Invoke-WebRequest -Uri $Url -OutFile $dst
    }
    return $dst
}

# 1) 임베디드 Python
$zip = Get-Cached "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip" "python-embed.zip"
Expand-Archive -Force $zip $Py
$pth = Get-ChildItem $Py -Filter "python*._pth" | Select-Object -First 1
# site-packages를 쓰도록 import site를 켠다
(Get-Content $pth.FullName) -replace "^#\s*import site", "import site" | Set-Content $pth.FullName -Encoding ascii

# 2) pip
$getpip = Get-Cached "https://bootstrap.pypa.io/get-pip.py" "get-pip.py"
& "$Py\python.exe" $getpip --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw "pip 설치 실패" }

# 3) 앱 휠과 고정된 의존성 설치(uv.lock 기준)
Push-Location $Root
uv build --wheel --out-dir (Join-Path $Build "wheel")
uv export --frozen --no-dev --no-hashes --no-emit-project -o (Join-Path $Build "requirements.txt")
Pop-Location
& "$Py\python.exe" -m pip install --no-warn-script-location -r (Join-Path $Build "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "의존성 설치 실패" }
$wheel = Get-ChildItem (Join-Path $Build "wheel") -Filter "*.whl" | Sort-Object LastWriteTime | Select-Object -Last 1
& "$Py\python.exe" -m pip install --no-warn-script-location --no-deps $wheel.FullName
if ($LASTEXITCODE -ne 0) { throw "앱 설치 실패" }

# PyTorch/TensorFlow가 끼어들지 않았는지 확인
$bad = & "$Py\python.exe" -m pip list --format=freeze | Select-String -Pattern "^(torch|tensorflow)=="
if ($bad) { throw "설치본에 들어가면 안 되는 패키지: $bad" }
& "$Py\python.exe" -c "import basic_pitch.inference, onnxruntime, gradio, acappella; print('import ok')"
if ($LASTEXITCODE -ne 0) { throw "import 확인 실패" }

# 4) ffmpeg
if (-not $FfmpegExe) {
    $fz = Get-Cached "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" "ffmpeg.zip"
    $fdir = Join-Path $Cache "ffmpeg"
    Expand-Archive -Force $fz $fdir
    $FfmpegExe = (Get-ChildItem $fdir -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1).FullName
}
Copy-Item $FfmpegExe (Join-Path $App "bin\ffmpeg.exe")

# 5) 모델(선택)
if ($ModelFile) { Copy-Item $ModelFile (Join-Path $App "models\voice_assign.onnx") }

# 6) 실행기: ffmpeg 경로를 설정으로 넘기고 앱을 연다
@"
@echo off
chcp 65001 > nul
set "ACAPPELLA_FFMPEG=%~dp0bin\ffmpeg.exe"
if not exist "%LOCALAPPDATA%\Acappella\models\voice_assign.onnx" if exist "%~dp0models\voice_assign.onnx" (
  mkdir "%LOCALAPPDATA%\Acappella\models" 2> nul
  copy /y "%~dp0models\voice_assign.onnx" "%LOCALAPPDATA%\Acappella\models\" > nul
)
echo 아카펠라 채보 앱을 시작합니다. 잠시 뒤 브라우저가 열립니다. 이 창을 닫으면 앱이 종료됩니다.
"%~dp0python\python.exe" -m acappella.app.ui
if errorlevel 1 pause
"@ | ForEach-Object {
    # .bat 파일은 BOM이 있으면 첫 줄이 깨진다: BOM 없는 UTF-8로 쓴다
    [System.IO.File]::WriteAllText((Join-Path $App "Acappella.bat"), ($_ -replace "`r?`n", "`r`n"), (New-Object System.Text.UTF8Encoding $false))
}

if ($SkipInstaller) { Write-Host "설치본 폴더: $App"; exit 0 }

# 7) Inno Setup
$version = (& "$Py\python.exe" -c "import acappella; print(acappella.__version__)").Trim()
$iscc = (Get-Command iscc -ErrorAction SilentlyContinue).Source
if (-not $iscc) { $iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" }
& $iscc "/DAppVersion=$version" "/DSourceDir=$App" "/O$Dist" (Join-Path $PSScriptRoot "installer.iss")
if ($LASTEXITCODE -ne 0) { throw "Inno Setup 실패" }
Write-Host "완료: $Dist"
