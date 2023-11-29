# Start from a base image that includes CUDA and PyTorch
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /main

# Copy the current directory contents into the container at /main
COPY . /main/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Define environment variable
ENV NAME World

# Run experiments.py when the container launches
CMD ["python", "experiments.py"]
