# AWS Lambda Deployment Guide

## Architecture

```
EventBridge (cron) ──► Lambda: tennis-predict-daily   (07:00 UTC daily)
                  ──► Lambda: tennis-clv-close         (4x daily, 30min before matches)
                  ──► Lambda: tennis-check-results     (23:30 UTC daily)
                  ──► Lambda: tennis-injury-refresh    (every 6 hours)
                  ──► Lambda: tennis-retrain           (Monday 03:00 UTC)

API Gateway ──────► Lambda: tennis-api                 (always-on REST endpoint)

All Lambdas ◄────── S3: tennis-odds-artifacts          (model files, data, logs)
```

## Cost (with $200 credit + free tier)

| Service | Free tier | Your usage | Monthly cost |
|---|---|---|---|
| Lambda compute | 400,000 GB-s/month | ~5,000 GB-s | **$0** |
| Lambda requests | 1M req/month | ~300 req | **$0** |
| S3 storage | 5 GB/month | ~2 GB | **$0** |
| S3 requests | 20,000 GET/month | ~5,000 | **$0** |
| API Gateway | 1M requests/month | ~100 | **$0** |
| ECR image | 500 MB/month | ~1 GB | ~$0.10 |
| **Total** | | | **~$0/month** |

Your $200 credit covers years of usage at this scale.

---

## Step-by-step: First Deploy

### Prerequisites (install once)

```bash
# macOS
brew install awscli aws-sam-cli docker

# Configure AWS credentials
aws configure
# Enter: Access Key ID, Secret Access Key, Region (e.g. ap-south-1), output format: json
```

Get your AWS credentials from: **IAM → Users → your user → Security credentials → Create access key**

### Step 1 — Train the model locally first

The model artifacts need to exist before deploying. Run this on your laptop:

```bash
uv run python main.py scrape
uv run python main.py features
uv run python main.py train
uv run python main.py tune-ensemble
uv run python main.py tune-thresholds
uv run python main.py calibrate
uv run python main.py sanity
```

### Step 2 — Set your API keys

```bash
export ODDS_API_KEY="your-odds-api-key"
export NOTIFY_SLACK_WEBHOOK="https://hooks.slack.com/..."   # optional
```

### Step 3 — Full deploy

```bash
chmod +x deploy/deploy.sh
./deploy/deploy.sh
```

This will:
1. Upload model artifacts to S3 (~2 min)
2. Build the Docker container and push to ECR (~5 min)
3. Deploy all Lambda functions + API Gateway via CloudFormation (~3 min)

At the end, you'll see the API URL printed.

### Step 4 — Test it

```bash
# Health check
curl https://<your-api-id>.execute-api.ap-south-1.amazonaws.com/prod/health

# Manual prediction
curl -X POST https://<your-api-id>.execute-api.ap-south-1.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -d '{"p1": "Sinner J.", "p2": "Alcaraz C.", "surface": "Clay"}'

# Trigger a prediction run manually (for testing)
aws lambda invoke \
  --function-name tennis-predict-daily \
  --region ap-south-1 \
  /tmp/response.json && cat /tmp/response.json
```

---

## Day-to-day operations

### Push code changes (no infra change)

```bash
./deploy/deploy.sh --update-code
```

### After retraining locally — sync new models to S3

```bash
./deploy/deploy.sh --upload-models
```

### Check Lambda logs

```bash
# Last 100 lines of daily prediction logs
aws logs tail /aws/lambda/tennis-predict-daily --since 24h

# CLV close logs
aws logs tail /aws/lambda/tennis-clv-close --since 24h
```

### Monitor scheduled jobs

```bash
# Check if jobs ran successfully (last 7 days)
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Errors \
  --dimensions Name=FunctionName,Value=tennis-predict-daily \
  --start-time $(date -u -v-7d +%Y-%m-%dT%H:%M:%SZ) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --period 86400 --statistics Sum
```

---

## Teardown (if you want to stop)

```bash
# Delete the whole stack (stops all billing)
aws cloudformation delete-stack --stack-name tennis-odds --region ap-south-1

# Delete ECR images
aws ecr delete-repository --repository-name tennis-odds --force --region ap-south-1

# Delete S3 bucket (empty it first)
aws s3 rm s3://tennis-odds-artifacts --recursive
aws s3 rb s3://tennis-odds-artifacts
```
