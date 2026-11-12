# Builds the Voice Changer Mic driver package and the CAB that Microsoft's
# attestation signing takes (Partner Center > Hardware > Submit new hardware).
#
#   powershell -ExecutionPolicy Bypass -File build.ps1 [-Configuration Release]
#
# Needs Visual Studio 2022 (or its Build Tools) with the Windows Driver Kit
# component and Spectre-mitigated libraries, and the WDK 10.0.26100.
param(
    [string]$Configuration = 'Release',
    [string]$Platform = 'x64'
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$msbuild = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -find 'MSBuild\**\Bin\MSBuild.exe' | Select-Object -First 1
if (-not $msbuild) { throw 'MSBuild not found; install Visual Studio 2022 Build Tools' }

# The in-build InfVerif task needs an x86 DLL that WDK 10.0.26100 no longer ships;
# infverif.exe runs below instead.
& $msbuild VirtualAudioDriver.sln /m /nologo /v:minimal `
    "/p:Configuration=$Configuration" "/p:Platform=$Platform" /p:SkipPackageVerification=true
if ($LASTEXITCODE -ne 0) { throw "Driver build failed ($LASTEXITCODE)" }

$package = Join-Path $PSScriptRoot "$Platform\$Configuration\package"
foreach ($file in 'VoiceChangerMic.inf', 'VoiceChangerMic.sys', 'voicechangermic.cat') {
    if (-not (Test-Path (Join-Path $package $file))) { throw "Missing $file in $package" }
}

$kits = "${env:ProgramFiles(x86)}\Windows Kits\10"
$infverif = Get-ChildItem "$kits\Tools" -Recurse -Filter infverif.exe |
    Where-Object { $_.FullName -match '\\x64\\' } | Sort-Object FullName -Descending | Select-Object -First 1
if (-not $infverif) { throw 'infverif.exe not found; install the WDK' }
# /w: the rules for a Windows Driver, the strictest set.
& $infverif.FullName /w /v (Join-Path $package 'VoiceChangerMic.inf')
if ($LASTEXITCODE -ne 0) { throw "infverif rejected the INF ($LASTEXITCODE)" }

# Attestation signing takes a CAB holding one folder per driver.
$dist = Join-Path $PSScriptRoot 'dist'
New-Item -ItemType Directory -Force $dist | Out-Null
$ddf = Join-Path $dist 'VoiceChangerMic.ddf'
@"
.OPTION EXPLICIT
.Set CabinetFileCountThreshold=0
.Set FolderFileCountThreshold=0
.Set FolderSizeThreshold=0
.Set MaxCabinetSize=0
.Set MaxDiskFileCount=0
.Set MaxDiskSize=0
.Set CompressionType=MSZIP
.Set Cabinet=on
.Set Compress=on
.Set DiskDirectoryTemplate="$dist"
.Set CabinetNameTemplate=VoiceChangerMic-$Platform.cab
.Set DestinationDir=VoiceChangerMic
"$package\VoiceChangerMic.inf"
"$package\VoiceChangerMic.sys"
"$package\voicechangermic.cat"
"@ | Set-Content -Encoding ascii $ddf
& makecab.exe /f $ddf | Out-Null
if ($LASTEXITCODE -ne 0) { throw "makecab failed ($LASTEXITCODE)" }
Remove-Item $ddf, (Join-Path $PSScriptRoot 'setup.inf'), (Join-Path $PSScriptRoot 'setup.rpt') -ErrorAction SilentlyContinue

Copy-Item (Join-Path $package '*') $dist -Force
Write-Host "Driver package and CAB in $dist"
