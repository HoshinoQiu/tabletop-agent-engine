param(
    [Parameter(Mandatory = $true)]
    [string]$BaseUrl,

    [string]$RulesDir = "data/rules",
    [string[]]$Extensions = @(".pdf", ".txt", ".md", ".markdown"),
    [int]$PollSeconds = 10,
    [switch]$SkipPoll,
    [ValidateSet("safe", "fast")]
    [string]$Mode = "safe"
)

$ErrorActionPreference = "Stop"

function Write-Info([string]$msg) {
    Write-Host "[INFO] $msg"
}

function Write-WarnMsg([string]$msg) {
    Write-Host "[WARN] $msg"
}

function Write-ErrMsg([string]$msg) {
    Write-Host "[ERR ] $msg"
}

function Parse-UploadResponse([string]$text) {
    $docId = $null
    $status = $null
    $filename = $null

    try {
        $obj = $text | ConvertFrom-Json -ErrorAction Stop
        $docId = $obj.doc_id
        $status = $obj.status
        $filename = $obj.filename
    } catch {
        if ($text -match '"doc_id"\s*:\s*"([^"]+)"') { $docId = $matches[1] }
        if ($text -match '"status"\s*:\s*"([^"]+)"') { $status = $matches[1] }
        if ($text -match '"filename"\s*:\s*"([^"]+)"') { $filename = $matches[1] }
    }

    return [PSCustomObject]@{
        doc_id = $docId
        status = $status
        filename = $filename
    }
}

$base = $BaseUrl.TrimEnd("/")
$uploadUrl = "$base/api/documents/upload"
$listUrl = "$base/api/documents"
$docStatusUrlTemplate = "$base/api/documents/{0}/status"

if (-not (Test-Path $RulesDir)) {
    throw "Rules directory not found: $RulesDir"
}

$extSet = @{}
foreach ($ext in $Extensions) {
    $e = $ext.ToLower()
    if (-not $e.StartsWith(".")) { $e = "." + $e }
    $extSet[$e] = $true
}

$files = Get-ChildItem -Path $RulesDir -File |
    Where-Object { $extSet.ContainsKey($_.Extension.ToLower()) } |
    Sort-Object Name

if (-not $files -or $files.Count -eq 0) {
    throw "No uploadable files found in $RulesDir"
}

Write-Info "Target: $base"
Write-Info "Files to upload: $($files.Count)"
Write-Info "Mode: $Mode"

$uploaded = 0
$failed = 0

foreach ($file in $files) {
    # Avoid duplicate uploads on reruns: skip if same filename is already processing/ready.
    try {
        $snapshot = Invoke-RestMethod -Method Get -Uri $listUrl
        $existing = @($snapshot.documents | Where-Object {
            $_.filename -eq $file.Name -and ($_.status -eq "processing" -or $_.status -eq "ready")
        })
        if ($existing.Count -gt 0) {
            Write-WarnMsg "Skip existing file: $($file.Name) (status=$($existing[0].status), doc_id=$($existing[0].doc_id))"
            continue
        }
    } catch {
        Write-WarnMsg "Could not check existing docs for $($file.Name), continue upload"
    }

    Write-Host ""
    Write-Info "Uploading: $($file.Name)"
    $ok = $false
    $lastErr = ""
    $uploadedDocId = $null
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            $respText = & curl.exe --http1.1 --retry 2 --retry-all-errors -sS -X POST "$uploadUrl" -F "file=@$($file.FullName)"
            if ($LASTEXITCODE -ne 0) {
                throw "curl exit code $LASTEXITCODE"
            }

            $resp = Parse-UploadResponse $respText
            if ($resp.status -eq "processing" -or $resp.status -eq "ready") {
                $uploaded++
                $ok = $true
                $uploadedDocId = $resp.doc_id
                Write-Info "Accepted doc_id=$($resp.doc_id) status=$($resp.status)"
                break
            }

            $lastErr = "unexpected response: $respText"
        } catch {
            $lastErr = $_.Exception.Message
        }

        if ($attempt -lt 3) {
            Write-WarnMsg "Retry $attempt failed for $($file.Name): $lastErr"
            Start-Sleep -Seconds 2
        }
    }

    if (-not $ok) {
        $failed++
        Write-ErrMsg "$($file.Name) -> $lastErr"
        continue
    }

    if ($Mode -eq "safe" -and $uploadedDocId) {
        Write-Info "Waiting doc_id=$uploadedDocId to finish before next upload..."
        while ($true) {
            try {
                $statusUrl = [string]::Format($docStatusUrlTemplate, $uploadedDocId)
                $doc = Invoke-RestMethod -Method Get -Uri $statusUrl
                $state = [string]$doc.status
                if ($state -eq "ready") {
                    Write-Info "done: $($file.Name) -> ready"
                    break
                }
                if ($state -eq "error") {
                    Write-WarnMsg "done: $($file.Name) -> error"
                    break
                }
                Write-Info "processing: $($file.Name) (doc_id=$uploadedDocId)"
            } catch {
                Write-WarnMsg "status check failed for $($file.Name): $($_.Exception.Message)"
            }
            Start-Sleep -Seconds $PollSeconds
        }
    }
}

Write-Host ""
Write-Info "Upload complete. success=$uploaded failed=$failed total=$($files.Count)"

if ($SkipPoll) {
    exit 0
}

Write-Info "Polling processing status every $PollSeconds seconds..."

while ($true) {
    try {
        $all = Invoke-RestMethod -Method Get -Uri $listUrl
        $docs = @($all.documents)
        $ready = @($docs | Where-Object { $_.status -eq "ready" }).Count
        $processing = @($docs | Where-Object { $_.status -eq "processing" }).Count
        $error = @($docs | Where-Object { $_.status -eq "error" }).Count

        Write-Info "ready=$ready processing=$processing error=$error total=$($docs.Count)"

        if ($processing -eq 0) {
            if ($error -gt 0) {
                Write-WarnMsg "Some files failed. Check /api/documents for details."
            } else {
                Write-Info "All files processed."
            }
            break
        }
    } catch {
        Write-ErrMsg "Status check failed: $($_.Exception.Message)"
    }

    Start-Sleep -Seconds $PollSeconds
}
