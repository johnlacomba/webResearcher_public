FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libxml2 \
    libxslt1.1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN playwright install --with-deps chromium

COPY research_tool/ research_tool/

RUN mkdir -p /root/.web_researcher/models /data

VOLUME ["/data", "/root/.web_researcher/models"]

ENTRYPOINT ["python3", "-m", "research_tool"]
CMD ["--help"]
