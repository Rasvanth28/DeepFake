# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV and ML
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the ML directory into the container
# We copy from the ml/ folder root
COPY . .

# Expose the port that Hugging Face expects (7860 is default)
EXPOSE 7860

# Command to run the application
# We need to specify the path to app.py relative to the ml/ folder
CMD ["uvicorn", "script.app:app", "--host", "0.0.0.0", "--port", "7860"]
