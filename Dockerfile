# Use official Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install build tools for native extensions
RUN apt-get update && \
    apt-get install -y build-essential g++ && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install uv
RUN pip install --upgrade pip && \
    pip install uv

# Install dependencies (from pyproject.toml or requirements.txt)
RUN uv pip install -r requirements.txt --system

# Expose Streamlit default port
EXPOSE 8501

# Set environment variable for Streamlit (optional)
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run Streamlit app
CMD ["streamlit", "run", "03_production_system/streamlit_app.py"]