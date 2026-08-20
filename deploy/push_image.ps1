<#
.SYNOPSIS
  Build the SentinelOps Lambda container image and push it to Amazon ECR.

.DESCRIPTION
  Wraps steps 1a-1d of deploy/AWS_DEPLOY.md. Safe to re-run: the ECR repository
  is only created if it does not already exist.

  Run from the REPO ROOT (the Docker build context must be the repo root):
      .\deploy\push_image.ps1 -Region ap-south-1

  After a successful push, update the function to pick up the new image:
      aws lambda update-function-code --function-name sentinelops-api `
        --image-uri <the URI printed at the end> --region ap-south-1

.PARAMETER Region
  AWS region. Must match the region of your Lambda, API Gateway and Amplify app.

.PARAMETER Repo
  ECR repository name.

.PARAMETER Tag
  Image tag. Defaults to 'latest'.
#>

param(
    [string]$Region = "ap-south-1",
    [string]$Repo   = "sentinelops-api",
    [string]$Tag    = "latest"
)

$ErrorActionPreference = "Stop"

# The build context must be the repo root — bail early rather than producing a
# confusing "COPY src/: not found" halfway through the build.
if (-not (Test-Path ".\Dockerfile")) {
    throw "Run this from the repository root (no Dockerfile in the current directory)."
}

Write-Host "==> Checking prerequisites" -ForegroundColor Cyan
docker info *> $null
if ($LASTEXITCODE -ne 0) { throw "Docker does not appear to be running. Start Docker Desktop and retry." }

$Account = (aws sts get-caller-identity --query Account --output text)
if ($LASTEXITCODE -ne 0) { throw "AWS CLI is not configured. Run 'aws configure' first." }

$Registry = "$Account.dkr.ecr.$Region.amazonaws.com"
$ImageUri = "$Registry/${Repo}:$Tag"
Write-Host "    account  $Account"
Write-Host "    region   $Region"
Write-Host "    image    $ImageUri"

Write-Host "==> Ensuring ECR repository exists" -ForegroundColor Cyan
aws ecr describe-repositories --repository-names $Repo --region $Region *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "    creating $Repo"
    aws ecr create-repository --repository-name $Repo --region $Region | Out-Null
} else {
    Write-Host "    already exists"
}

Write-Host "==> Logging Docker in to ECR" -ForegroundColor Cyan
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $Registry
if ($LASTEXITCODE -ne 0) { throw "docker login to ECR failed." }

# --platform is not optional: Lambda runs x86_64, and Docker on an ARM machine
# would otherwise build an arm64 image that fails at runtime with 'exec format
# error' — an error that gives no hint about its actual cause.
Write-Host "==> Building image (first build takes 5-10 min)" -ForegroundColor Cyan
docker build --platform linux/amd64 -t "${Repo}:$Tag" .
if ($LASTEXITCODE -ne 0) { throw "docker build failed." }

Write-Host "==> Tagging and pushing" -ForegroundColor Cyan
docker tag "${Repo}:$Tag" $ImageUri
docker push $ImageUri
if ($LASTEXITCODE -ne 0) { throw "docker push failed." }

Write-Host ""
Write-Host "Pushed: $ImageUri" -ForegroundColor Green
Write-Host ""
Write-Host "Next:" -ForegroundColor Yellow
Write-Host "  New function      -> Lambda console, Create function, Container image, browse to the tag above"
Write-Host "  Existing function -> aws lambda update-function-code --function-name $Repo --image-uri $ImageUri --region $Region"
Write-Host "  Then confirm memory 1024 MB / timeout 30 s (see deploy/AWS_DEPLOY.md step 2)."
