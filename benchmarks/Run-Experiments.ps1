param(
    [ValidateSet('local-sequential','local-parallel','distributed','parallel-distributed')]
    [string[]]$Scenario = @('local-sequential','local-parallel'),
    [ValidateRange(1,100)][int]$Repetitions = 3,
    [ValidateRange(1,1000000)][int]$Requests = 300,
    [ValidateRange(1,256)][int]$Clients = 1,
    [ValidateRange(0,1000000)][int]$Warmup = 100,
    [long]$Seed = 42,
    [ValidateRange(0,60000)][int]$MinDelayMs = 1,
    [ValidateRange(0,60000)][int]$MaxDelayMs = 3,
    [ValidateRange(1,300000)][int]$TimeoutMs = 15000,
    [ValidateRange(2,32)][int]$ParallelCores = 2,
    [string]$CoordinatorHost = '192.168.1.11',
    [int]$CoordinatorPort = 9000,
    [string]$Agents = '192.168.1.11:9200,192.168.1.12:9200,192.168.1.13:9200',
    [string]$OutputDirectory = 'build/stage6/measurements'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($MaxDelayMs -lt $MinDelayMs) { throw 'MaxDelayMs < MinDelayMs' }
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
foreach ($profile in $Scenario) {
    foreach ($repeat in 1..$Repetitions) {
        $destination = Join-Path $OutputDirectory "$profile-c$Clients-r$repeat.csv"
        if (Test-Path -LiteralPath $destination) { throw "Result already exists: $destination. Choose a new OutputDirectory." }
        $baseline = Join-Path $OutputDirectory "local-sequential-c$Clients-r$repeat.csv.summary.properties"
        $arguments = @('-cp','build','benchmarks.Benchmark','--config',"benchmarks/configs/$profile.properties",
            "requests=$Requests","clients=$Clients","warmup=$Warmup","seed=$Seed","repetition=$repeat",
            "delay.min.ms=$MinDelayMs","delay.max.ms=$MaxDelayMs","timeout.ms=$TimeoutMs",
            "output=$destination","baseline=$baseline","host=$CoordinatorHost","port=$CoordinatorPort","agents=$Agents")
        if ($profile -eq 'local-parallel') { $arguments += @("cores=$ParallelCores","threads=$ParallelCores") }
        & java @arguments
        if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $profile repetition $repeat" }
    }
}
