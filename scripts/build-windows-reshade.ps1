# Build openOMSI for Windows into dist\windows as scripts\build-windows.cmd does, with the one
# change that lets ReShade load into the game:
#
#   wgpu (the project's fork, openOMSI-Project/wgpu-openomsi) makes an IDXGIFactoryMedia through
#   the legacy CreateDXGIFactory1 when it opens DirectX 12. ReShade 6.8 hooks that export and
#   wraps the factory it hands out as an ordinary DXGI factory, and the process went down with
#   an access violation inside dxgi.dll before any window opened. openOMSI never makes the
#   surfaces that factory is for (DirectComposition handles), and wgpu treats it as optional
#   (`.ok()`), so the build leaves it out.
#
# The source is not kept in this repository: the script copies the fork at the revision
# Cargo.lock names out of Cargo's own checkout into %LOCALAPPDATA%\openOMSI\reshade-build
# (outside the repository: under it, Cargo took the copy's crates for members of openOMSI's
# own workspace), takes the one line out there and builds with every crate of the fork
# patched to that copy (one crate alone would leave two wgpu-hal and two wgpu-types in the
# build). When a newer fork no longer has the
# line, the script stops and says so - a stale workaround never builds quietly.
#
# Cargo.lock is left as it was: the build writes the copy's paths into it, and the file is put
# back afterwards.
#
#   powershell -ExecutionPolicy Bypass -File scripts\build-windows-reshade.ps1
#
# Needs what build-windows.cmd needs (Rust with the MSVC toolchain, Visual Studio Build Tools).
# x64 only: ReShade's DirectX 12 add-ons are for x64.

param([string]$Target = "x86_64-pc-windows-msvc")

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$root = (Get-Location).Path
$fork = "https://github.com/openOMSI-Project/wgpu-openomsi"
$work = Join-Path $root "target\reshade"
New-Item -ItemType Directory -Force $work | Out-Null

# the fork's crates as Cargo.lock has them
$lockText = [IO.File]::ReadAllText((Join-Path $root "Cargo.lock"))
$found = [regex]::Matches($lockText, '(?m)^name = "([^"]+)"\r?\nversion = "[^"]+"\r?\nsource = "git\+' + [regex]::Escape($fork) + '\?rev=[0-9a-f]+#([0-9a-f]+)"')
$names = @($found | ForEach-Object { $_.Groups[1].Value })
$revs = @($found | ForEach-Object { $_.Groups[2].Value } | Sort-Object -Unique)
if ($names -notcontains "wgpu-hal") { throw "wgpu-hal does not come from $fork any more (Cargo.lock) - the workaround has to be looked at again" }
if ($revs.Count -ne 1) { throw "Cargo.lock names the fork at more than one revision: $($revs -join ', ')" }
$rev = $revs[0]
Write-Host "wgpu fork at $rev ($($names.Count) crates: $($names -join ', '))"

# where Cargo checked it out (CARGO_HOME\git\checkouts\wgpu-openomsi-<hash>\<short revision>)
& cargo fetch --locked
if ($LASTEXITCODE -ne 0) { throw "cargo fetch failed" }
$cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE ".cargo" }
$checkout = Get-ChildItem (Join-Path $cargoHome "git\checkouts") -Directory -Filter "wgpu-openomsi-*" |
    ForEach-Object { Join-Path $_.FullName $rev.Substring(0, 7) } |
    Where-Object { Test-Path (Join-Path $_ ".cargo-ok") } | Select-Object -First 1
if (-not $checkout) { throw "no Cargo checkout of $fork at $rev under $cargoHome\git\checkouts" }
# each crate's folder in it, by the name in its manifest
$manifests = Get-ChildItem $checkout -Recurse -Filter Cargo.toml -File | Where-Object { $_.FullName -notmatch '\\(target|\.git)\\' }
$crates = foreach ($n in $names) {
    $m = $manifests | Where-Object { [IO.File]::ReadAllText($_.FullName) -match ('(?ms)^\[package\][^\[]*?^name = "' + [regex]::Escape($n) + '"') } | Select-Object -First 1
    if (-not $m) { throw "no crate $n in $checkout" }
    [pscustomobject]@{ name = $n; dir = $m.DirectoryName }
}

# the copy, made once a revision
$copy = Join-Path $env:LOCALAPPDATA "openOMSI\reshade-build\wgpu-openomsi-$rev"
if (-not (Test-Path (Join-Path $copy ".reshade-ok"))) {
    if (Test-Path $copy) { Remove-Item -Recurse -Force $copy }
    # (robocopy: 1 means copied; 8 and above is a failure)
    & robocopy $checkout $copy /E /XD .git /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "copying the fork failed (robocopy $LASTEXITCODE)" }
    $file = Join-Path $copy "wgpu-hal\src\dx12\instance.rs"
    $text = [IO.File]::ReadAllText($file)
    $old = "let factory_media = lib_dxgi.create_factory_media().ok();"
    if (-not $text.Contains($old)) {
        throw "wgpu-hal at ${rev}: '$old' is not in src\dx12\instance.rs any more - see what the fork does with IDXGIFactoryMedia now before building for ReShade"
    }
    $new = "// openOMSI (scripts\build-windows-reshade.ps1): no IDXGIFactoryMedia - ReShade wraps the`n" +
           "        // factory CreateDXGIFactory1 hands out for it, and the process went down in dxgi.dll`n" +
           "        let factory_media: Option<Dxgi::IDXGIFactoryMedia> = None;`n" +
           "        let _ = &lib_dxgi;"
    [IO.File]::WriteAllText($file, $text.Replace($old, $new))
    # (the method is unused now: an allowance instead of a warning on every build)
    $mod = Join-Path $copy "wgpu-hal\src\dx12\mod.rs"
    $modText = [IO.File]::ReadAllText($mod)
    $fn = "    pub fn create_factory_media("
    if ($modText.Contains($fn)) {
        [IO.File]::WriteAllText($mod, $modText.Replace($fn, "    #[allow(dead_code)]`n$fn"))
    }
    New-Item -ItemType File (Join-Path $copy ".reshade-ok") | Out-Null
}

# every crate of the fork from the copy, for both ways it is asked for (crates.io's name in
# the project's [patch.crates-io], and the fork's own git address between its crates)
$lines = @()
foreach ($table in @("patch.crates-io", "patch.`"$fork`"")) {
    $lines += "[$table]"
    foreach ($c in $crates) {
        $rel = $c.dir.Substring($checkout.Length).TrimStart("\")
        $path = (Join-Path $copy $rel) -replace "\\", "/"
        $lines += "$($c.name) = { path = `"$path`" }"
    }
}
$config = Join-Path $work "patch.toml"
[IO.File]::WriteAllText($config, ($lines -join "`n") + "`n")

$lock = Join-Path $work "Cargo.lock.orig"
Copy-Item Cargo.lock $lock -Force
try {
    & cargo build --release --target $Target --config $config -p omsi-app -p omsi-launcher-core
    if ($LASTEXITCODE -ne 0) { throw "the build failed (see above)" }
} finally {
    Copy-Item $lock Cargo.lock -Force
}

# the program into dist\windows, as build-windows.cmd puts it there
$dist = Join-Path $root "dist\windows"
New-Item -ItemType Directory -Force $dist | Out-Null
Copy-Item "target\$Target\release\openomsi.exe", "target\$Target\release\openomsi-launcher.exe" $dist -Force
if ($Target -eq "x86_64-pc-windows-msvc") {
    Copy-Item "assets\steam_redist\steam_api64.dll" $dist -Force
}
Write-Host ""
Write-Host "Done (with the ReShade workaround). Run: $dist\openomsi.exe"
