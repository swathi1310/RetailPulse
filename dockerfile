FROM apache/airflow:latest

# Switch to root user to install system packages
USER root

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    gfortran \
    liblapack-dev \
    git && \
    apt-get clean

# Copy the correct requirements.txt file to the container
COPY ./requirements.txt /tmp/requirements.txt

# Switch to airflow user
USER airflow

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt