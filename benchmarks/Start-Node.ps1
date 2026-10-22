param(
    [Parameter(Mandatory=$true)][ValidateSet('A','B','C')][string]$Role,
    [Parameter(Mandatory=$true)][ValidateSet('DISTRIBUTED','PARALLEL_DISTRIBUTED')][string]$Scenario,
    [ValidateRange(2,32)][int]$Cores = 2,
    [string]$LaptopA = '192.168.1.11',
    [string]$LaptopB = '192.168.1.12',
    [string]$LaptopC = '192.168.1.13'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$budget = if ($Scenario -eq 'DISTRIBUTED') { 1 } else { $Cores }
$addresses = @{ A=$LaptopA; B=$LaptopB; C=$LaptopC }
$languages = @{ A='Java'; B='CSharp'; C='Python' }
$ports = @{ A=9101; B=9102; C=9103 }
$arguments = @('benchmarks/node_agent.py','--scenario',$Scenario,'--language',$languages[$Role],
    '--cores',"$budget",'--worker-port',"$($ports[$Role])",'--agent-port','9200','--advertise',$addresses[$Role],
    '--logs',"build/stage6/nodes/$Scenario-$Role")
if ($Role -eq 'A') {
    $workers = @("${LaptopA}:9101","${LaptopB}:9102")
    foreach ($index in 0..($budget-1)) { $workers += "${LaptopC}:$(9103+$index)" }
    $arguments += @('--coordinator-port','9000','--coordinator-workers')+$workers
}
& python @arguments
if ($LASTEXITCODE -ne 0) { throw "Node failed: $Role $Scenario" }
