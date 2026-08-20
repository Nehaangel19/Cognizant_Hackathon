<#
.SYNOPSIS
  Create (or update) the SentinelOps Lambda function and its API Gateway HTTP API.

.DESCRIPTION
  Automates steps 2 and 3 of deploy/AWS_DEPLOY.md. Run it AFTER deploy/push_image.ps1
  has pushed an image to ECR. Safe to re-run: existing role / function / API are
  reused and updated rather than duplicated.

      .\deploy\create_lambda.ps1 -Region ap-south-1

  On success it prints the public Invoke URL. That URL is what goes into Amplify's
  VITE_API_BASE environment variable.

.PARAMETER Region
  Must match the region the image was pushed to.
#>

param(
    [string]$Region       = "ap-south-1",
    [string]$FunctionName = "sentinelops-api",
    [string]$Repo         = "sentinelops-api",
    [string]$Tag          = "latest",
    [string]$RoleName     = "sentinelops-lambda-role"
)

# See push_image.ps1 for why this is not "Stop" — the AWS CLI writes progress and
# warnings to stderr, which PowerShell 5.1 would otherwise treat as fatal.
$ErrorActionPreference = "Continue"

function Fail($msg) { Write-Host $msg -ForegroundColor Red; exit 1 }

Write-Host "==> Resolving account and image" -ForegroundColor Cyan
$Account = (aws sts get-caller-identity --query Account --output text 2>&1)
if ($LASTEXITCODE -ne 0) { Fail "AWS CLI not authenticated. Run 'aws login' (or 'aws configure') first.`n$Account" }

$ImageUri = "$Account.dkr.ecr.$Region.amazonaws.com/${Repo}:$Tag"
aws ecr describe-images --repository-name $Repo --image-ids imageTag=$Tag --region $Region 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "Image not found: $ImageUri`nRun .\deploy\push_image.ps1 -Region $Region first." }
Write-Host "    $ImageUri"

# --- IAM execution role ---------------------------------------------------
Write-Host "==> Ensuring execution role" -ForegroundColor Cyan
$RoleArn = (aws iam get-role --role-name $RoleName --query "Role.Arn" --output text 2>&1)
if ($LASTEXITCODE -ne 0) {
    # Written to a temp file rather than passed inline: PowerShell's handling of
    # embedded double quotes in native-command arguments mangles inline JSON.
    $trust = @'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}
'@
    $trustFile = Join-Path $env:TEMP "sentinelops-trust.json"
    $trust | Set-Content -Path $trustFile -Encoding ASCII

    $RoleArn = (aws iam create-role --role-name $RoleName `
                  --assume-role-policy-document "file://$trustFile" `
                  --query "Role.Arn" --output text 2>&1)
    if ($LASTEXITCODE -ne 0) { Fail "Could not create role.`n$RoleArn" }

    aws iam attach-role-policy --role-name $RoleName `
        --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole" 2>&1 | Out-Null

    Write-Host "    created $RoleName"
    # IAM is eventually consistent. Creating a function with a role that exists
    # but has not propagated yet fails with an unhelpful InvalidParameterValue.
    Write-Host "    waiting 15s for IAM propagation..."
    Start-Sleep -Seconds 15
} else {
    Write-Host "    reusing $RoleName"
}

# --- Lambda function ------------------------------------------------------
Write-Host "==> Creating / updating function" -ForegroundColor Cyan
aws lambda get-function --function-name $FunctionName --region $Region 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    $out = (aws lambda create-function --function-name $FunctionName `
              --package-type Image --code "ImageUri=$ImageUri" --role $RoleArn `
              --architectures x86_64 --memory-size 1024 --timeout 30 `
              --region $Region --output text 2>&1)
    if ($LASTEXITCODE -ne 0) { Fail "create-function failed.`n$out" }
    Write-Host "    created $FunctionName"
} else {
    Write-Host "    exists - updating code and config"
    aws lambda update-function-code --function-name $FunctionName `
        --image-uri $ImageUri --region $Region 2>&1 | Out-Null
    aws lambda wait function-updated --function-name $FunctionName --region $Region 2>&1 | Out-Null
    aws lambda update-function-configuration --function-name $FunctionName `
        --memory-size 1024 --timeout 30 --region $Region 2>&1 | Out-Null
}
aws lambda wait function-active-v2 --function-name $FunctionName --region $Region 2>&1 | Out-Null

$FnArn = (aws lambda get-function --function-name $FunctionName --region $Region `
            --query "Configuration.FunctionArn" --output text 2>&1)

# --- API Gateway HTTP API -------------------------------------------------
Write-Host "==> Ensuring HTTP API" -ForegroundColor Cyan
$ApiId = (aws apigatewayv2 get-apis --region $Region `
            --query "Items[?Name=='$FunctionName'].ApiId | [0]" --output text 2>&1)

if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($ApiId) -or $ApiId -eq "None") {
    # --target does the whole wiring in one call: AWS_PROXY integration with
    # payload format 2.0, a $default catch-all route, and an auto-deploying
    # $default stage. FastAPI does its own routing, so a catch-all is what we want.
    $ApiId = (aws apigatewayv2 create-api --name $FunctionName --protocol-type HTTP `
                --target $FnArn --region $Region --query "ApiId" --output text 2>&1)
    if ($LASTEXITCODE -ne 0) { Fail "create-api failed.`n$ApiId" }
    Write-Host "    created API $ApiId"
} else {
    Write-Host "    reusing API $ApiId"
}

# Without this, every request returns 500 with an opaque permissions error.
# Re-running is harmless: a duplicate statement id just errors and is ignored.
aws lambda add-permission --function-name $FunctionName --statement-id apigw-invoke `
    --action lambda:InvokeFunction --principal apigateway.amazonaws.com `
    --source-arn "arn:aws:execute-api:${Region}:${Account}:${ApiId}/*/*" `
    --region $Region 2>&1 | Out-Null

$InvokeUrl = "https://$ApiId.execute-api.$Region.amazonaws.com"

# --- Smoke test -----------------------------------------------------------
Write-Host "==> Smoke testing (first call is a cold start, allow ~6s)" -ForegroundColor Cyan
Start-Sleep -Seconds 3
try {
    $health = Invoke-RestMethod -Uri "$InvokeUrl/health" -TimeoutSec 40 -UseBasicParsing
    Write-Host "    status       $($health.status)"
    Write-Host "    modelsLoaded $($health.modelsLoaded)"
    Write-Host "    totalCycles  $($health.totalCycles)"
    if (-not $health.modelsLoaded) {
        Write-Host "    WARNING: models did not load. Check CloudWatch logs:" -ForegroundColor Yellow
        Write-Host "      aws logs tail /aws/lambda/$FunctionName --region $Region --since 5m"
    }
} catch {
    Write-Host "    smoke test failed: $_" -ForegroundColor Yellow
    Write-Host "    check logs: aws logs tail /aws/lambda/$FunctionName --region $Region --since 5m"
}

Write-Host ""
Write-Host "API URL: $InvokeUrl" -ForegroundColor Green
Write-Host ""
Write-Host "Next:" -ForegroundColor Yellow
Write-Host "  1. Amplify -> Environment variables -> VITE_API_BASE = $InvokeUrl"
Write-Host "  2. After Amplify gives you its URL, allow it through CORS:"
Write-Host "     aws lambda update-function-configuration --function-name $FunctionName ``"
Write-Host "       --environment `"Variables={ALLOWED_ORIGINS=https://main.XXXX.amplifyapp.com}`" --region $Region"
