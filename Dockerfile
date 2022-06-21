FROM herrd1/siquan:latest


WORKDIR /energy

# install some packages to reduce download 
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

RUN pip install "pyomo>=5.7,<=6.0"

COPY requirements.txt /energy/requirements.txt
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


ENTRYPOINT [ "python3", "run.py"]
