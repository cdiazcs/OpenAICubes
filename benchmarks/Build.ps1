$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$javaSources = @(Get-ChildItem clients-java,frontend-java,server-java,benchmarks,workers/java -Filter *.java -Recurse | ForEach-Object FullName)
New-Item -ItemType Directory -Force build | Out-Null
& javac --release 8 -encoding UTF-8 -d build @javaSources
if ($LASTEXITCODE -ne 0) { throw 'Java compilation failed' }
& dotnet build workers/csharp/Worker.csproj -o build/csharp --nologo
if ($LASTEXITCODE -ne 0) { throw 'CSharp compilation failed' }
& python benchmarks/prepare_inputs.py
if ($LASTEXITCODE -ne 0) { throw 'Input preparation failed' }
