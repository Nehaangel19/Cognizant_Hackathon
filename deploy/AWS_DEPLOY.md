# Deploying SentinelOps on AWS

Target architecture:

```
  Browser
     │
     ▼
  AWS Amplify Hosting          ← React/Vite dashboard (static)
     │  fetch(VITE_API_BASE)
     ▼
  API Gateway (HTTP API)       ← public HTTPS endpoint, $default stage
     │  AWS_PROXY integration
     ▼
  Lambda (container image)     ← FastAPI via Mangum, 1024 MB, 30 s timeout
     │                              models bundled in the image (default)
     └── S3 (optional)         ← or pulled from a bucket at cold start
```

Everything below sits inside the AWS Free Tier at demo traffic. See **Cost** at the
end for the actual numbers and the one line item that is not free.

**Set your region once and use it everywhere.** These docs use `ap-south-1`
(Mumbai). Amplify, Lambda, ECR and API Gateway must all be in the same region or
you will spend an hour on a "why can't it find the image" error.

---

## 0. Prerequisites

| Need | Check | Notes |
|---|---|---|
| AWS account | — | Card required for verification; nothing here should bill |
| AWS CLI v2 | `aws --version` | `aws configure` with an access key |
| Docker Desktop | `docker --version` | Must be **running** before step 1 |
| Repo pushed to GitHub | `git status` | Amplify deploys from GitHub, not from disk |

Confirm the CLI is pointed where you think it is:

```powershell
aws sts get-caller-identity
aws configure get region
```

---

## 1. Build and push the Lambda image to ECR

`deploy/push_image.ps1` does all of this. Run it from the repo root:

```powershell
.\deploy\push_image.ps1 -Region ap-south-1
```

Or the same steps by hand:

```powershell
$REGION  = "ap-south-1"
$ACCOUNT = (aws sts get-caller-identity --query Account --output text)
$REPO    = "sentinelops-api"

# 1a. Create the ECR repository (once)
aws ecr create-repository --repository-name $REPO --region $REGION

# 1b. Log Docker in to ECR
aws ecr get-login-password --region $REGION | `
  docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

# 1c. Build. Context is the REPO ROOT, not deploy/.
#     --platform matters: Lambda runs x86_64 and Docker on an ARM Mac
#     would otherwise build an arm64 image that fails at runtime with an
#     opaque "exec format error".
docker build --platform linux/amd64 -t $REPO .

# 1d. Tag and push
docker tag "${REPO}:latest" "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/${REPO}:latest"
docker push "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/${REPO}:latest"
```

First build takes 5-10 minutes (xgboost, shap and scikit-learn wheels are large).
Rebuilds after a code-only change are much faster — the dependency layer is cached
above the `COPY src/` line specifically so that editing Python does not reinstall
the ML stack.

**Test the image locally before pushing it anywhere.** The Lambda base image ships
the Runtime Interface Emulator, so you can invoke it exactly as Lambda would:

```powershell
docker run --rm -p 9000:8080 sentinelops-api

# in a second terminal
curl -s "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{\"version\":\"2.0\",\"rawPath\":\"/health\",\"rawQueryString\":\"\",\"headers\":{},\"isBase64Encoded\":false,\"requestContext\":{\"http\":{\"method\":\"GET\",\"path\":\"/health\",\"sourceIp\":\"127.0.0.1\",\"protocol\":\"HTTP/1.1\",\"userAgent\":\"curl\"},\"stage\":\"$default\",\"requestId\":\"r1\",\"apiId\":\"a1\",\"domainName\":\"local\",\"accountId\":\"1\",\"time\":\"01/Jan/2026:00:00:00 +0000\",\"timeEpoch\":0}}'
```

Expect `{"statusCode":200, ... "modelsLoaded":true ...}`. If `modelsLoaded` is
`false`, the models did not make it into the image — check `models/*.joblib` exist
locally and are not excluded by `.dockerignore`.

---

## 2. Create the Lambda function

Console → **Lambda** → Create function → **Container image**.

| Field | Value |
|---|---|
| Function name | `sentinelops-api` |
| Container image URI | Browse images → `sentinelops-api:latest` |
| Architecture | `x86_64` |

Then **Configuration → General configuration → Edit**:

| Setting | Value | Why |
|---|---|---|
| Memory | **1024 MB** | Memory scales CPU on Lambda. At 512 MB the xgboost/shap import alone pushes cold start past 10 s; 1024 MB is *cheaper per request* despite the higher per-ms rate because it finishes more than twice as fast. |
| Timeout | **30 s** | Cold start is ~3-6 s. The 3 s default will time out on the first request and look like a broken deploy. |
| Ephemeral storage | 512 MB (default) | Only matters if you use the S3 model option, which writes to `/tmp`. |

CLI equivalent:

```powershell
aws lambda update-function-configuration `
  --function-name sentinelops-api `
  --memory-size 1024 --timeout 30 --region ap-south-1
```

**Test it in the console** before wiring API Gateway. Test tab → Event JSON:

```json
{"version":"2.0","rawPath":"/health","rawQueryString":"","headers":{},"isBase64Encoded":false,
 "requestContext":{"http":{"method":"GET","path":"/health","sourceIp":"127.0.0.1","protocol":"HTTP/1.1","userAgent":"test"},
 "stage":"$default","requestId":"r1","apiId":"a1","domainName":"local","accountId":"1","time":"01/Jan/2026:00:00:00 +0000","timeEpoch":0}}
```

---

## 3. Create the API Gateway endpoint

Console → **API Gateway** → **HTTP API** (not REST API — HTTP APIs are simpler and
cheaper, and 1M calls/month are free) → Build.

1. **Integrations**: Lambda → `sentinelops-api`. Leave *payload format version* at
   **2.0** — `src/api/lambda_handler.py` is built against the v2.0 event shape.
2. **Routes**: add a single catch-all route — method `ANY`, path `/{proxy+}`.
   FastAPI does its own routing; API Gateway just needs to hand everything over.
   Add a second `ANY /` route if you want the bare root to resolve too.
3. **Stages**: keep `$default` with auto-deploy on. A named stage prefixes every
   path (`/prod/health`), which then has to be stripped — avoid it.

Copy the **Invoke URL** (`https://<api-id>.execute-api.ap-south-1.amazonaws.com`).
Verify:

```powershell
curl https://<api-id>.execute-api.ap-south-1.amazonaws.com/health
```

The first call takes several seconds (cold start), subsequent ones are fast.

> **Do not** enable API Gateway's own CORS configuration. FastAPI's
> `CORSMiddleware` already sets those headers, and both layers setting them
> produces a duplicate `Access-Control-Allow-Origin` header that browsers reject.
> One or the other — we use FastAPI's, configured in step 6.

---

## 4. (Optional) Serve models from S3

**You do not need this.** The five `.joblib` artefacts total ~7.5 MB and are baked
into the image, well under Lambda's 10 GB limit. Skip to step 5 unless you
specifically want the decoupling — in which case it is a real architectural
improvement worth explaining: retraining becomes "re-upload five files", not
"rebuild and redeploy a 1.2 GB image".

```powershell
$BUCKET = "sentinelops-models-<your-suffix>"   # bucket names are globally unique
aws s3 mb "s3://$BUCKET" --region ap-south-1
aws s3 cp models/ "s3://$BUCKET/models/" --recursive --exclude "*" --include "*.joblib"
```

Grant the function's execution role read access — Lambda console → Configuration →
Permissions → click the role → Add permissions → Create inline policy → JSON:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:GetObject"],
    "Resource": "arn:aws:s3:::sentinelops-models-<your-suffix>/models/*"
  }]
}
```

Then set on the function (Configuration → Environment variables):

| Key | Value |
|---|---|
| `MODEL_S3_BUCKET` | `sentinelops-models-<your-suffix>` |
| `MODEL_S3_PREFIX` | `models` |

`src/api/model_store.py` downloads to `/tmp` once per cold start and reuses it
while the container stays warm. If the fetch fails for any reason it logs why and
falls back to the bundled copy rather than returning 500s — a demo on
slightly-stale models beats a demo that is down.

---

## 5. Deploy the frontend on Amplify

Console → **Amplify** → Create new app → **GitHub** → authorise → pick
`Nehaangel19/Cognizant_Hackathon`, branch `main`.

Amplify should detect `amplify.yml` at the repo root and show the `frontend`
monorepo build spec. If it asks whether this is a monorepo, say yes and set the
app root to `frontend`.

Before the first build, add the environment variable (Hosting → Environment
variables):

| Key | Value |
|---|---|
| `VITE_API_BASE` | `https://<api-id>.execute-api.ap-south-1.amazonaws.com` |

No trailing slash. Vite inlines this at **build** time, so if you change it later
you must **Redeploy this version** — restarting is not enough.

Save and deploy. You get `https://main.<app-id>.amplifyapp.com`.

---

## 6. Wire CORS (the step everyone forgets)

The browser now loads from an `amplifyapp.com` origin and calls an
`execute-api.amazonaws.com` origin. That is cross-origin, so the API must name the
frontend explicitly.

Lambda console → Configuration → Environment variables → add:

| Key | Value |
|---|---|
| `ALLOWED_ORIGINS` | `https://main.<app-id>.amplifyapp.com` |

Comma-separate if you have several (a custom domain, a preview branch). Localhost
origins are always allowed, so local development keeps working unchanged.

Saving env vars restarts the function — the next request is a cold start.

---

## 7. Verify end to end

```powershell
$API = "https://<api-id>.execute-api.ap-south-1.amazonaws.com"

curl "$API/health"                     # modelsLoaded: true, totalCycles: 10000
curl "$API/demo-cases"                 # the four seeded demo modes
curl "$API/reading/70"                 # CRITICAL / HIGH / OSF
curl "$API/reading/78"                 # WARNING / CONFLICT / null  ← the "won't bluff" case
curl -X POST "$API/analyze" -H "Content-Type: application/json" -d '{\"cycleId\":70,\"mode\":\"offline\"}'
```

Then open the Amplify URL and walk all four tabs. Browser devtools → Network: the
requests should go to `execute-api`, return 200, and show no CORS errors in the
console.

**Warm it up before you present.** Hit `/health` a few minutes before the demo so
the first judge-facing click is not a 4-second cold start.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `exec format error` in Lambda logs | Image built for arm64 | Rebuild with `--platform linux/amd64` |
| 502 Bad Gateway from API Gateway | Handler crashed on import | CloudWatch → `/aws/lambda/sentinelops-api` → read the traceback |
| `/health` says `modelsLoaded: false` | Artefacts missing from image | Check `models/*.joblib` exist locally, rebuild, re-push |
| Task timed out after 3.00 seconds | Timeout never raised | Step 2 — set it to 30 s |
| CORS error in browser console | `ALLOWED_ORIGINS` unset or has a typo/trailing slash | Step 6 — must match the Amplify origin exactly, scheme included |
| Duplicate `Access-Control-Allow-Origin` | CORS enabled on API Gateway *and* FastAPI | Disable it on API Gateway |
| Frontend calls `/api/...` and 404s | `VITE_API_BASE` unset at build time | Set it, then **Redeploy this version** |
| Amplify build fails on `npm ci` | `package-lock.json` out of sync with `package.json` | `npm install` locally, commit the lockfile |
| Pushing a new image does not change behaviour | Lambda pins the image digest | Lambda console → Deploy new image, or `aws lambda update-function-code --image-uri ...` |

Logs live in CloudWatch → Log groups → `/aws/lambda/sentinelops-api`. The cold
start line `[lambda] cold start init finished in N.NNs` and
`[api] models loaded — threshold 0.2486` are the two to look for.

---

## Cost

| Service | Free tier | This project |
|---|---|---|
| Lambda | 1M requests + 400,000 GB-s / month | A demo is a few thousand requests. Free. |
| API Gateway (HTTP API) | 1M calls / month for 12 months | Free. |
| Amplify Hosting | Build minutes + GB served, free-tier allowance | A handful of builds. Free. |
| S3 | 5 GB standard storage | 7.5 MB. Free. |
| **ECR** | **500 MB/month private storage** | **This image is ~1.2 GB.** |

ECR is the one line to watch: private-registry storage above 500 MB is billed
(~$0.10/GB-month), so a 1.2 GB image costs a few cents per month, not zero. To
avoid even that, delete old image tags after each push — ECR keeps every untagged
layer otherwise:

```powershell
aws ecr list-images --repository-name sentinelops-api --filter tagStatus=UNTAGGED --query 'imageIds[*]' --output json |
  aws ecr batch-delete-image --repository-name sentinelops-api --image-ids file:///dev/stdin
```

**Tear down after the hackathon** so nothing lingers: delete the Amplify app, the
API, the Lambda function, the ECR repository, and the S3 bucket if you made one.
