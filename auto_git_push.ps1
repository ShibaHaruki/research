$ErrorActionPreference = "Stop"

$RepoPath = $PSScriptRoot
$LogDir = Join-Path $PSScriptRoot "git_log"
$LogPath = Join-Path $LogDir "auto_git_push.log"

if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
}

Set-Location $RepoPath

$Date = Get-Date -Format "yyyy-MM-dd"
$Now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

function Write-Log($Message) {
    $Time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$Time $Message" | Add-Content -Path $LogPath
}

function Run-Git {
    param([string[]]$GitArgs)

    $Output = & git @GitArgs 2>&1

    if ($LASTEXITCODE -ne 0) {
        Write-Log "git $($GitArgs -join ' ') failed"
        Write-Log $Output
        throw "git command failed: git $($GitArgs -join ' ')"
    }

    return $Output
}

Write-Log "start"

$Branch = (& git branch --show-current).Trim()

$Changes = git status --porcelain

if ([string]::IsNullOrWhiteSpace($Changes)) {
    Write-Log "no changes"
    exit 0
}

Run-Git @("add", "-A")
Run-Git @("commit", "-m", $Date)
Run-Git @("push", "origin", $Branch)

Write-Log "commit and push complete: $Date"

