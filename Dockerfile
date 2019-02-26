# Use an official Python runtime as a parent image
FROM frolvlad/alpine-miniconda3

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./src/ /app/src

# Install any needed packages
RUN . activate base
# RUN conda install 
RUN pip install numpy pandas sklearn tqdm

# Run unit tests
RUN python -m unittest discover -s ./src -p '*_tests.py'

# Run script when the container launches
ENTRYPOINT ["python", "./src/sat.py"]