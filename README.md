TODO -> 

## Build the container
```
docker build . -t energy:0.1
```

## Run the container
```
docker run --mount type=bind,source=/path/to/Problemset,target=/energy/Problemset energy:0.1 dwave-tabu
```

## Parameters
Using environment variables. For the possibilities see `run.py`.
```
docker run --mount type=bind,source=/path/to/Problemset,target=/energy/Problemset --env trotterSlices=32 energy:0.1 sqa
```