FROM herrd1/siquan:latest


WORKDIR /energy
COPY DockerInput/requirements.txt /energy/requirements.txt
RUN pip install -r /energy/requirements.txt
RUN apt-get install -y glpk-utils 

COPY DockerInput /energy
RUN chmod -R u+wxr /energy

# add placeholder for the input model
RUN mkdir /energy/input-model
RUN chmod u+xr /energy/input-model


ENTRYPOINT [ "python3", "run.py"]
