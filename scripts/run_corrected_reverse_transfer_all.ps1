[CmdletBinding()]
param(
    [switch]$CommitRegistry,
    [switch]$PlanOnly,
    [switch]$DryRunOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$WrapperVersion = "corrected_reverse_transfer_orchestrator_v1"
$FixedConfigDirectory = "configs\corrected_lendingclub_to_homecredit"
$FixedOutputDirectory = "results\corrected_lendingclub_to_homecredit_transfer"
$FixedSeeds = "11,22,33,44,55"
$FixedModels = "lr,catboost"
$MainCliRelativePath = "scripts\run_corrected_lendingclub_to_homecredit_transfer.py"

function New-ReverseTransferContract {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepositoryRoot
    )

    $pythonInterpreter = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
    $mainScript = Join-Path $RepositoryRoot $MainCliRelativePath
    $configDirectory = Join-Path $RepositoryRoot $FixedConfigDirectory
    $outputDirectory = Join-Path $RepositoryRoot $FixedOutputDirectory

    return [pscustomobject]@{
        RepositoryRoot = $RepositoryRoot
        PythonInterpreter = $pythonInterpreter
        MainScript = $mainScript
        ConfigDirectory = $configDirectory
        OutputDirectory = $outputDirectory
        RequiredConfigFiles = @(
            (Join-Path $configDirectory "contrastive_data.yaml"),
            (Join-Path $configDirectory "training.yaml"),
            (Join-Path $configDirectory "reverse_projection.yaml"),
            (Join-Path $configDirectory "downstream.yaml"),
            (Join-Path $configDirectory "identity_evidence.json")
        )
        RequiredRawInputs = @(
            (Join-Path $RepositoryRoot "data\lendingclub_v2\processed\application_train.csv"),
            (Join-Path $RepositoryRoot "data\homecredit\raw\application_train.csv")
        )
    }
}

function New-StageDefinition {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$LogFileName,

        [Parameter(Mandatory = $true)]
        [string]$Stage,

        [switch]$DryRun
    )

    $arguments = @(
        $MainCliRelativePath,
        "--stage", $Stage,
        "--config-dir", $FixedConfigDirectory,
        "--output-dir", $FixedOutputDirectory,
        "--seeds", $FixedSeeds,
        "--models", $FixedModels
    )
    if ($DryRun) {
        $arguments += "--dry-run"
    }

    return [pscustomobject]@{
        Name = $Name
        LogFileName = $LogFileName
        Arguments = $arguments
        DisplayCommand = ".\.venv\Scripts\python.exe " + ($arguments -join " ")
    }
}

function Get-ReverseTransferStages {
    return @(
        (New-StageDefinition -Name "01_preflight" -LogFileName "01_preflight.log" -Stage "all" -DryRun),
        (New-StageDefinition -Name "02_prepare" -LogFileName "02_prepare.log" -Stage "prepare"),
        (New-StageDefinition -Name "03_train" -LogFileName "03_train.log" -Stage "train"),
        (New-StageDefinition -Name "04_project" -LogFileName "04_project.log" -Stage "project"),
        (New-StageDefinition -Name "05_evaluate" -LogFileName "05_evaluate.log" -Stage "evaluate"),
        (New-StageDefinition -Name "06_register_dry_run" -LogFileName "06_register_dry_run.log" -Stage "register" -DryRun),
        (New-StageDefinition -Name "07_register_commit" -LogFileName "07_register_commit.log" -Stage "register")
    )
}

function Assert-ReverseTransferPrerequisites {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Contract,

        [Parameter(Mandatory = $true)]
        [object[]]$Stages
    )

    if (-not (Test-Path -LiteralPath $Contract.PythonInterpreter -PathType Leaf)) {
        throw "Required virtual-environment interpreter is missing: $($Contract.PythonInterpreter)"
    }
    if (-not (Test-Path -LiteralPath $Contract.MainScript -PathType Leaf)) {
        throw "Required reverse-transfer CLI script is missing: $($Contract.MainScript)"
    }
    if (-not (Test-Path -LiteralPath $Contract.ConfigDirectory -PathType Container)) {
        throw "Required reverse-transfer configuration directory is missing: $($Contract.ConfigDirectory)"
    }
    foreach ($path in $Contract.RequiredConfigFiles) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required reverse-transfer configuration file is missing: $path"
        }
    }
    foreach ($path in $Contract.RequiredRawInputs) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required raw-data input is missing: $path"
        }
    }

    $expectedOutput = [IO.Path]::GetFullPath(
        (Join-Path $Contract.RepositoryRoot $FixedOutputDirectory)
    )
    $actualOutput = [IO.Path]::GetFullPath($Contract.OutputDirectory)
    if (-not $actualOutput.Equals($expectedOutput, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Output directory is not the fixed reverse-transfer output root: $actualOutput"
    }

    $forbiddenPatterns = @(
        "run_corrected_homecredit_clip_pipelines",
        "train_clip_encoder",
        "run_clip_final_comparison",
        "clip_final_comparison",
        "umap",
        "stable_core",
        "baseline",
        "task_1",
        "task_3",
        "matrix"
    )
    foreach ($stage in $Stages) {
        $command = $stage.DisplayCommand.ToLowerInvariant()
        foreach ($pattern in $forbiddenPatterns) {
            if ($command.Contains($pattern)) {
                throw "Forbidden unrelated command detected in $($stage.Name): $pattern"
            }
        }
    }
}

function Invoke-ReverseTransferStage {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Stage,

        [Parameter(Mandatory = $true)]
        [string]$PythonInterpreter,

        [Parameter(Mandatory = $true)]
        [string]$LogDirectory,

        [Parameter(Mandatory = $true)]
        [hashtable]$State
    )

    $logPath = Join-Path $LogDirectory $Stage.LogFileName
    $started = [DateTimeOffset]::UtcNow
    $record = [ordered]@{
        stage_name = $Stage.Name
        command = $Stage.DisplayCommand
        start_timestamp = $started.ToString("o")
        end_timestamp = $null
        exit_code = $null
        log_path = $logPath
    }
    $State.ActiveStage = $Stage.Name

    Write-Host ""
    Write-Host "[$($Stage.Name)] $($Stage.DisplayCommand)"
    Write-Host "Log: $logPath"

    $stageOutput = [System.Collections.ArrayList]::new()
    $exitCode = 1
    $utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
    $logWriter = [System.IO.StreamWriter]::new(
        $logPath,
        $false,
        $utf8WithoutBom
    )
    try {
        & $PythonInterpreter @($Stage.Arguments) 2>&1 |
            ForEach-Object {
                $line = $_.ToString()
                [void]$stageOutput.Add($line)
                $logWriter.WriteLine($line)
                $logWriter.Flush()
                Write-Host $line
            }
        $exitCode = $LASTEXITCODE
    }
    finally {
        $logWriter.Dispose()
        $record.end_timestamp = [DateTimeOffset]::UtcNow.ToString("o")
        $record.exit_code = $exitCode
        [void]$State.StageResults.Add([pscustomobject]$record)
    }

    if ($exitCode -ne 0) {
        $State.FailedStage = $Stage.Name
        throw "Stage '$($Stage.Name)' failed with exit code $exitCode. Log: $logPath"
    }

    return [pscustomobject]@{
        StageName = $Stage.Name
        ExitCode = $exitCode
        LogPath = $logPath
        OutputText = @($stageOutput) -join [Environment]::NewLine
    }
}

function Assert-RegistryDryRunApproved {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$StageResult
    )

    try {
        $payload = $StageResult.OutputText | ConvertFrom-Json
    }
    catch {
        throw "Registry dry-run output was not valid JSON. Log: $($StageResult.LogPath)"
    }

    if ($null -eq $payload.registry_dry_run) {
        throw "Registry dry-run output omitted registry validation results. Log: $($StageResult.LogPath)"
    }

    $validation = $payload.registry_dry_run
    $outcome = [string]$validation.transaction_outcome
    $acceptableOutcomes = @("NEW_TRANSACTION", "IDEMPOTENT_NO_OP")
    $missingArtifacts = @($validation.missing_artifacts)

    if ($validation.writes_performed -ne $false) {
        throw "Registry dry-run unexpectedly reported writes. Log: $($StageResult.LogPath)"
    }
    if ($validation.success_transaction_manifest_written -ne $false) {
        throw "Registry dry-run unexpectedly wrote a transaction manifest. Log: $($StageResult.LogPath)"
    }
    if ($outcome -eq "CONFLICT") {
        throw "Registry dry-run reported CONFLICT. Registry commit is blocked. Log: $($StageResult.LogPath)"
    }
    if ($missingArtifacts.Count -gt 0) {
        throw "Registry dry-run reported missing artifacts. Registry commit is blocked. Log: $($StageResult.LogPath)"
    }
    if ($acceptableOutcomes -notcontains $outcome) {
        throw "Registry dry-run outcome '$outcome' is not acceptable. Registry commit is blocked. Log: $($StageResult.LogPath)"
    }

    return $outcome
}

function Write-OrchestrationSummary {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SummaryPath,

        [Parameter(Mandatory = $true)]
        [hashtable]$State,

        [Parameter(Mandatory = $true)]
        [pscustomobject]$Contract,

        [Parameter(Mandatory = $true)]
        [object[]]$Stages,

        [Parameter(Mandatory = $true)]
        [string]$RequestedMode
    )

    $summary = [ordered]@{
        wrapper_version = $WrapperVersion
        start_timestamp = $State.StartTimestamp
        end_timestamp = [DateTimeOffset]::UtcNow.ToString("o")
        repository_root = $Contract.RepositoryRoot
        python_interpreter = $Contract.PythonInterpreter
        configuration_directory = $FixedConfigDirectory
        output_directory = $FixedOutputDirectory
        seeds = $FixedSeeds
        models = $FixedModels
        requested_mode = $RequestedMode
        stage_names = @($Stages | ForEach-Object { $_.Name })
        commands = @($Stages | ForEach-Object { $_.DisplayCommand })
        stage_results = @($State.StageResults)
        overall_status = $State.OverallStatus
        failed_stage = $State.FailedStage
        registry_dry_run_passed = $State.RegistryDryRunPassed
        registry_dry_run_outcome = $State.RegistryDryRunOutcome
        registry_commit_attempted = $State.RegistryCommitAttempted
        registry_commit_occurred = $State.RegistryCommitOccurred
    }
    $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $SummaryPath -Encoding UTF8
}

function Show-ExecutionHeader {
    param(
        [bool]$RegistryCommitEnabled
    )

    Write-Host "Corrected reverse transfer"
    Write-Host "Source: LendingClub v2"
    Write-Host "External dataset: Home Credit"
    Write-Host "Seeds: 11,22,33,44,55"
    Write-Host "Models: LR, CatBoost"
    Write-Host "Output root: results/corrected_lendingclub_to_homecredit_transfer"
    Write-Host "Registry commit enabled: $($RegistryCommitEnabled.ToString().ToLowerInvariant())"
}

function Show-CompletionSummary {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$State,

        [Parameter(Mandatory = $true)]
        [string]$LogDirectory
    )

    Write-Host ""
    Write-Host "overall status: $($State.OverallStatus)"
    Write-Host "completed stages: $((@($State.StageResults) | Where-Object { $_.exit_code -eq 0 } | ForEach-Object { $_.stage_name }) -join ', ')"
    Write-Host "output root: $FixedOutputDirectory"
    Write-Host "orchestration log directory: $LogDirectory"
    Write-Host "registry dry-run passed: $($State.RegistryDryRunPassed)"
    Write-Host "registry commit occurred: $($State.RegistryCommitOccurred)"
    Write-Host "next step: run Prompt 3 post-run audit"
}

function Invoke-CorrectedReverseTransfer {
    $selectedModes = @($CommitRegistry, $PlanOnly, $DryRunOnly | Where-Object { $_ }).Count
    if ($selectedModes -gt 1) {
        [Console]::Error.WriteLine(
            "-CommitRegistry, -PlanOnly, and -DryRunOnly are mutually exclusive."
        )
        return 2
    }

    $repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
    $contract = New-ReverseTransferContract -RepositoryRoot $repositoryRoot
    $stages = @(Get-ReverseTransferStages)
    $requestedMode = if ($PlanOnly) {
        "plan_only"
    }
    elseif ($DryRunOnly) {
        "dry_run_only"
    }
    elseif ($CommitRegistry) {
        "full_with_registry_commit"
    }
    else {
        "scientific_without_registry_commit"
    }

    Show-ExecutionHeader -RegistryCommitEnabled ([bool]$CommitRegistry)

    try {
        Assert-ReverseTransferPrerequisites -Contract $contract -Stages $stages
    }
    catch {
        [Console]::Error.WriteLine($_.Exception.Message)
        return 3
    }

    if ($PlanOnly) {
        Write-Host ""
        Write-Host "Validated execution plan (no stages will run):"
        foreach ($stage in $stages) {
            Write-Host "$($stage.Name): $($stage.DisplayCommand)"
        }
        Write-Host ""
        Write-Host "Plan-only complete. No scientific command or registry command was executed."
        return 0
    }

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
    $logDirectory = Join-Path $contract.OutputDirectory "orchestration_logs\$timestamp"
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $summaryPath = Join-Path $logDirectory "orchestration_summary.json"
    $state = @{
        StartTimestamp = [DateTimeOffset]::UtcNow.ToString("o")
        StageResults = [System.Collections.ArrayList]::new()
        ActiveStage = $null
        FailedStage = $null
        OverallStatus = "in_progress"
        RegistryDryRunPassed = $false
        RegistryDryRunOutcome = $null
        RegistryCommitAttempted = $false
        RegistryCommitOccurred = $false
    }
    $wrapperExitCode = 0

    Push-Location $repositoryRoot
    try {
        if ($DryRunOnly) {
            [void](Invoke-ReverseTransferStage -Stage $stages[0] -PythonInterpreter $contract.PythonInterpreter -LogDirectory $logDirectory -State $state)
            $registryResult = Invoke-ReverseTransferStage -Stage $stages[5] -PythonInterpreter $contract.PythonInterpreter -LogDirectory $logDirectory -State $state
            $state.RegistryDryRunOutcome = Assert-RegistryDryRunApproved -StageResult $registryResult
            $state.RegistryDryRunPassed = $true
            $state.OverallStatus = "completed_dry_run_only"
        }
        else {
            foreach ($stage in $stages[0..4]) {
                [void](Invoke-ReverseTransferStage -Stage $stage -PythonInterpreter $contract.PythonInterpreter -LogDirectory $logDirectory -State $state)
            }

            $registryResult = Invoke-ReverseTransferStage -Stage $stages[5] -PythonInterpreter $contract.PythonInterpreter -LogDirectory $logDirectory -State $state
            $state.RegistryDryRunOutcome = Assert-RegistryDryRunApproved -StageResult $registryResult
            $state.RegistryDryRunPassed = $true

            if ($CommitRegistry) {
                $state.RegistryCommitAttempted = $true
                [void](Invoke-ReverseTransferStage -Stage $stages[6] -PythonInterpreter $contract.PythonInterpreter -LogDirectory $logDirectory -State $state)
                $state.RegistryCommitOccurred = $true
                $state.OverallStatus = "completed_and_registered"
            }
            else {
                $state.OverallStatus = "completed_not_registered"
                Write-Host ""
                Write-Host "Scientific outputs were generated but were not registered."
                Write-Host "Re-run with -CommitRegistry only after reviewing the completed outputs and dry-run evidence."
            }
        }
    }
    catch {
        $state.OverallStatus = "failed"
        if ($null -eq $state.FailedStage) {
            $state.FailedStage = $state.ActiveStage
        }
        $wrapperExitCode = 1
        [Console]::Error.WriteLine($_.Exception.Message)
        if ($null -ne $state.FailedStage) {
            [Console]::Error.WriteLine("Failed stage: $($state.FailedStage)")
            $failedRecord = @($state.StageResults | Where-Object { $_.stage_name -eq $state.FailedStage } | Select-Object -Last 1)
            if ($failedRecord.Count -gt 0) {
                [Console]::Error.WriteLine("Relevant log: $($failedRecord[0].log_path)")
            }
        }
    }
    finally {
        Pop-Location
        try {
            Write-OrchestrationSummary -SummaryPath $summaryPath -State $state -Contract $contract -Stages $stages -RequestedMode $requestedMode
        }
        catch {
            $wrapperExitCode = 1
            [Console]::Error.WriteLine("Failed to write orchestration summary: $($_.Exception.Message)")
        }
        Show-CompletionSummary -State $state -LogDirectory $logDirectory
    }

    return $wrapperExitCode
}

if ($MyInvocation.InvocationName -ne ".") {
    exit (Invoke-CorrectedReverseTransfer)
}
