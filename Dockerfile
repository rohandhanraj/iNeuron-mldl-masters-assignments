# Base image
FROM python:3.7
# Add all the files to the present working directory
COPY . /app
# Define the present working directory
WORKDIR /app
# Install pip requirements
RUN pip install -r requirements.txt
# Set the entrypoint to the main.py file
ENTRYPOINT [ "python" ]
# Execute the command 
CMD [ "main.py" ]