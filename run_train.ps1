param(
    [int]$Seed = 61,
    [Nullable[int]]$MaskSeed = $null,
    [string]$Dataname = 'BDGP',
    [double]$MissingRate = 0.5
)

$env:PATH = 'D:\Anaconda\envs\lycenv\Library\bin;D:\Anaconda\envs\lycenv\Scripts;D:\Anaconda\envs\lycenv;C:\Windows\System32'
$env:PYTHONPATH = 'D:\Anaconda\envs\lycenv\Lib\site-packages'
$env:LOKY_MAX_CPU_COUNT = '8'

Set-Location -LiteralPath $PSScriptRoot
$trainArgs = @(
    '-S', 'train.py',
    '--dataname', $Dataname,
    '--seed', $Seed,
    '--missing-rate', $MissingRate
)
if ($null -ne $MaskSeed) {
    $trainArgs += @('--mask-seed', $MaskSeed)
}
& 'D:\Anaconda\python.exe' @trainArgs

if ($LASTEXITCODE -ne 0) {
    throw "CausalMVC clean training failed with exit code $LASTEXITCODE"
}
