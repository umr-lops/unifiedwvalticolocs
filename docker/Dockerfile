FROM python:3.11-slim-bookworm

ENV LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PATH="${PATH}:/root/.poetry/bin"

# Install system deps
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false

WORKDIR /workspace

# 1. Copy ONLY dependency files first (better Docker caching)
COPY pyproject.toml poetry.lock* ./

# 2. Install dependencies (but NOT the project yet, because code isn't copied)
RUN poetry install --no-root --without dev

# 3. Copy the rest of your source code
COPY . .

# 4. Install YOUR project (this creates the executable and makes it importable)
# We accept the dev arg here to decide if we want dev tools, but we MUST install the root project now.
ARG INSTALL_DEV=false
RUN bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install ; else poetry install --without dev ; fi"

# (Optional) Verify installation during build
RUN which procunifiedwvalticolocs

CMD mkdir -p /workspace
