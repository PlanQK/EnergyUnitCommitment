# This dockerfile builds the image for running the optimization of the unit commitment
# problem. The container is started using a makefile and the image rebuild using this 
# file if docker.tmp is out of date

# contains compiled siquan binaries
FROM herrd1/siquan:latest

WORKDIR /energy

# Install some packages to reduce download size when remaking the images
# The installed packaged are all in the requirements.txt
RUN pip install pyomo==6.4.1 && \
    pip install pypsa==0.20.0 && \
    pip install numpy==1.23.1 && \
    pip install scipy==1.8.1 && \
    pip install pandas==1.4.3 && \
    pip install matplotlib==3.5.2 
RUN pip install dimod==0.11.2 && \
    pip install qiskit==0.36.2 && \
    pip install networkx==2.8.4 && \
    pip install tables==3.7.0 && \
    pip install minorminer==0.2.9

COPY requirements.txt /energy/requirements.txt

# temporary until pypsa on pip has fix for reading serialized networks
# COPY pypsa-0.19.3.zip /energy/pypsa-0.19.3.zip
# RUN pip install pypsa-0.19.3.zip

RUN pip install -r /energy/requirements.txt
RUN apt-get install -y glpk-utils 
COPY src /energy
RUN chmod -R u+wxr /energy

# add placeholder for the input model
RUN mkdir /energy/input-model
RUN chmod u+xr /energy/input-model
# Flag for showing full stack trace if running it locally by reraising errors
ENV RUNNING_IN_DOCKER Yes
# Flag for diabling runtime limitations
ENV TRUSTED_USER Yes

ENTRYPOINT [ "python3", "run.py"]
