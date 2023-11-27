# Use a base image with your desired Python version
FROM python:3.8

# Set the working directory in the container
WORKDIR /experiments

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run experiments.py when the container launches
CMD ["python", "experiments.py"]
