FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install OS-level dependencies required by some Python packages and Streamlit
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	   build-essential \
	   libgl1 \
	&& rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run Streamlit in headless/server mode
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
