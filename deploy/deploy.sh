#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Full deploy: build container → push to ECR → deploy SAM stack
#
# Usage:
#   ./deploy/deploy.sh                         # first deploy
#   ./deploy/deploy.sh --update-code           # just push new code, no infra change
#   ./deploy/deploy.sh --upload-models         # just sync model artifacts to S3
#
# Prerequisites:
#   brew install awscli aws-sam-cli docker
#   aws configure   (set your region + credentials)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

AWS_REGION="${AWS_DEFAULT_REGION:-ap-south-1}"   # ← change to your region
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="tennis-odds"
ECR_URI="$AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO"
S3_BUCKET="${S3_BUCKET:-tennis-odds-artifacts}"
STACK_NAME="tennis-odds"

MODE="${1:-}"

echo "=================================================="
echo " Tennis Odds — AWS Lambda Deploy"
echo " Region : $AWS_REGION"
echo " Account: $AWS_ACCOUNT"
echo "=================================================="

# ── Upload model artifacts to S3 ─────────────────────────────────────────────
upload_models() {
    echo ""
    echo "[S3] Uploading model artifacts..."
    if [ ! -d "data/processed" ]; then
        echo "  ERROR: data/processed not found. Run the training pipeline first."
        exit 1
    fi

    # Create bucket if it doesn't exist
    aws s3 mb "s3://$S3_BUCKET" --region "$AWS_REGION" 2>/dev/null || true

    aws s3 sync data/processed/ "s3://$S3_BUCKET/data/processed/" \
        --exclude "*.pyc" --exclude "__pycache__/*" \
        --delete
    echo "  Done. Artifacts at s3://$S3_BUCKET/data/processed/"
}

# ── Build and push Docker image ───────────────────────────────────────────────
build_and_push() {
    echo ""
    echo "[Docker] Building image..."
    docker build -f deploy/Dockerfile -t "$ECR_REPO:latest" .

    echo "[ECR] Logging in..."
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin \
        "$AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com"

    # Create ECR repo if missing
    aws ecr describe-repositories --repository-names "$ECR_REPO" \
        --region "$AWS_REGION" 2>/dev/null || \
    aws ecr create-repository --repository-name "$ECR_REPO" \
        --region "$AWS_REGION" --image-scanning-configuration scanOnPush=true

    echo "[ECR] Pushing image..."
    docker tag "$ECR_REPO:latest" "$ECR_URI:latest"
    docker push "$ECR_URI:latest"
    echo "  Image: $ECR_URI:latest"
}

# ── Deploy SAM stack ──────────────────────────────────────────────────────────
deploy_stack() {
    echo ""
    echo "[SAM] Deploying stack $STACK_NAME..."

    # Prompt for secrets if not in env
    ODDS_KEY="${ODDS_API_KEY:-}"
    SLACK_URL="${NOTIFY_SLACK_WEBHOOK:-}"

    if [ -z "$ODDS_KEY" ]; then
        read -r -p "  Enter ODDS_API_KEY: " ODDS_KEY
    fi

    sam deploy \
        --template-file deploy/template.yaml \
        --stack-name "$STACK_NAME" \
        --region "$AWS_REGION" \
        --capabilities CAPABILITY_IAM \
        --parameter-overrides \
            "S3BucketName=$S3_BUCKET" \
            "OddsApiKey=$ODDS_KEY" \
            "SlackWebhook=$SLACK_URL" \
        --image-repository "$ECR_URI" \
        --no-confirm-changeset

    echo ""
    echo "[SAM] Stack outputs:"
    aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$AWS_REGION" \
        --query "Stacks[0].Outputs" \
        --output table
}

# ── Route based on mode ───────────────────────────────────────────────────────
case "$MODE" in
    --upload-models)
        upload_models
        ;;
    --update-code)
        build_and_push
        # Force Lambda to pull the new image
        for fn in tennis-predict-daily tennis-clv-close tennis-check-results \
                  tennis-injury-refresh tennis-retrain tennis-api; do
            echo "  Updating $fn..."
            aws lambda update-function-code \
                --function-name "$fn" \
                --image-uri "$ECR_URI:latest" \
                --region "$AWS_REGION" > /dev/null
        done
        echo "All functions updated."
        ;;
    *)
        # Full deploy
        upload_models
        build_and_push
        deploy_stack
        ;;
esac

echo ""
echo "=================================================="
echo " Deploy complete!"
echo " Test the API:"
API_URL=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" --region "$AWS_REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" \
    --output text 2>/dev/null || echo "<run full deploy to get URL>")
echo "   curl $API_URL/health"
echo "=================================================="
