# SAT-solver
UvA 2019 Information Retrieval assignment

## Run with docker
```
# build
docker build -t sat -f ./Dockerfile .

# run
docker run -v <absolut-path-to-data-directory>:/data sat /data sudoku-example-processed.txt -S2

# save image
docker save sat -o ./sat

# load image
docker load -i ./sat
```

## Run with python
```
python ./src/sat.py ./data/sudoku-example-processed.txt -S2
```

## Development

Run all unit tests
```
python -m unittest discover -s ./src -p '*_tests.py'
```
