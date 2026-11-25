# MLOps Demo: End-to-End Commands

`train.py → model.pkl → app.py (API) → Docker → GitHub Actions (CI)`

## 1. Setup
```bash
cd ~/Desktop/mlops/hello-world-mlops
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Train the model (Data scientist)
```bash
python train.py                 # 80/20 split, trains, saves artifacts/model.pkl
```

## 3. Run the API (ML engineer)
```bash
python app.py                   #
```
Test in Terminal 2:
```bash
curl localhost:5001/health
curl -X POST localhost:5001/predict -H "Content-Type: application/json" -d '{"features":[6.0,2.9,4.5,1.5]}'
# {"flower":"versicolor","prediction":1}
```


## 4. Run in Docker (MLOps)
```bash
docker build -t iris-api .         # build image, name it "iris-api"
docker run -p 5001:5001 iris-api   # start a container from it
docker ps        # containers running right now
docker ps -a     # all containers, including stopped ones
docker images    # your images (iris-api is here)
```
Test in Terminal 2 (same commands):
```bash
curl localhost:5001/health
curl -X POST localhost:5001/predict -H "Content-Type: application/json" -d '{"features":[6.0,2.9,4.5,1.5]}'
```
Stop with **Ctrl+C**.

## 5. CI/CD (MLOps)
```bash
git add . && git commit -m "demo update" && git push
adding new version 
``` 