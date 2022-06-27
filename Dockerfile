# This dockerfile builds the image for running the optimization of the unit commitment
# problem. The container is started using a makefile and the image rebuild using this 
# file if docker.tmp is out of date

# contains compiled siquan binaries
FROM herrd1/siquan:latest

WORKDIR /energy

# Install some packages to reduce download size when remaking the images
# The installed packaged are all in the requirements.txt
RUN pip install numpy && \
    pip install scipy && \
    pip install pandas && \
    pip install matplotlib && \
    pip install pyomo==6.0 && \
    pip install dimod && \
    pip install qiskit && \
    pip install networkx && \
    pip install tables && \
    pip install minorminer

COPY requirements.txt /energy/requirements.txt

# temporary until pypsa on pip has fix for reading serialized networks
COPY pypsa-0.19.3.zip /energy/pypsa-0.19.3.zip
RUN pip install pypsa-0.19.3.zip

RUN pip install -r /energy/requirements.txt
RUN apt-get install -y glpk-utils 
COPY src /energy
RUN chmod -R u+wxr /energy

# add placeholder for the input model
RUN mkdir /energy/input-model
RUN chmod u+xr /energy/input-model
ENV RUNNING_IN_DOCKER Yes
ENV TRUSTED_USER Yes

ENTRYPOINT [ "python3", "run.py"]
