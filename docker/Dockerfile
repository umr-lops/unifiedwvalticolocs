FROM python:3.14-slim-bookworm

# 1. Declare Arguments and Environment Variables
ARG CI_JOB_TOKEN
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# 2. Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Poetry
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false

# 4. Configure Poetry for private GitLab registry (Cleanest way)
# This replaces your manual "pip install s1ifr"
RUN poetry config repositories.ifremer https://gitlab.ifremer.fr/api/v4/projects/4991/packages/pypi/simple && \
    poetry config http-basic.ifremer gitlab-ci-token ${CI_JOB_TOKEN}

# 5. Copy dependency files
COPY pyproject.toml poetry.lock* ./

# 6. Install dependencies
# We use a build arg to decide whether to include dev dependencies
ARG INSTALL_DEV=false
RUN if [ "$INSTALL_DEV" = "true" ] ; then poetry install --no-root ; else poetry install --no-root --without dev ; fi

# 7. Copy the rest of the source code
COPY . .

# 8. Install the project itself
# We remove any existing dist folder copied from host,
# then build a fresh wheel and install only that.
# RUN rm -rf dist && \
#     poetry build -f wheel && \
#     pip install dist/*.whl

# 8. Install the project itself (the "root" package)
# This installs your package and its entry points (procunifiedwvalticolocs)
RUN if [ "$INSTALL_DEV" = "true" ] ; then poetry install ; else poetry install --without dev ; fi

# 9. Verify installation
RUN which procunifiedwvalticolocs

# 10. The command to run your app
CMD ["python", "-m", "unifiedwvalticolocs"]
