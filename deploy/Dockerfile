# Stage 1: grab libgomp from a full Amazon Linux image
FROM --platform=linux/amd64 amazonlinux:2 AS system-libs
RUN yum install -y libgomp && yum clean all

# Stage 2: Lambda runtime
FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.11

# Copy libgomp from the full image
COPY --from=system-libs /usr/lib64/libgomp.so.1 /usr/lib64/libgomp.so.1

WORKDIR ${LAMBDA_TASK_ROOT}

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/            ./src/
COPY tools/          ./tools/
COPY main.py         ./
COPY deploy/lambda_handler.py ./deploy/
COPY scripts/        ./scripts/

# Install deps — use only pre-built wheels (no compilation)
RUN pip install --no-cache-dir --only-binary :all: \
    lightgbm \
    pandas \
    numpy \
    scikit-learn \
    requests \
    fastapi \
    "uvicorn[standard]" \
    mangum \
    openpyxl \
    "pyarrow>=14.0,<16.0" \
    boto3 \
    beautifulsoup4 \
    lxml

CMD ["deploy.lambda_handler.dispatch"]
