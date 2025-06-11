# Use the official Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure Uvicorn is installed explicitly
RUN pip install fastapi uvicorn

# Expose the FastAPI port
EXPOSE 7860

# Run FastAPI with Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]