# MLOps Demo: End-to-End Commands (one sheet)

**Flow:** `Data → train.py → model.pkl → run_model.py → app.py (API) → Docker → git push (CI)`

---

## 0. One-time setup (already done on this laptop)
```bash
sudo apt install -y docker.io            # install Docker
sudo usermod -aG docker $USER            # allow docker without sudo
# then LOG OUT and LOG IN once
```

## 1. Go to project + Python environment
```bash
cd ~/Desktop/mlops/hello-world-mlops
python3 -m venv .venv                    # create virtual env (first time only)
source .venv/bin/activate                # activate it
pip install -r requirements.txt          # install flask, scikit-learn, joblib, pandas
```

## 2. Train the model  ← *Data scientist part*
```bash
python train.py
```
Expected:
```
Train rows: 120 | Test rows: 30          ← 80/20 split
[6.1, 2.8, 4.7, 1.2]   versicolor   versicolor   yes
...
Saved model to artifacts/model.pkl       ← packaged model
Test accuracy: 1.0000
```
```bash
ls artifacts/                            # model.pkl  metrics.json
cat artifacts/metrics.json               # {"accuracy": 1.0}
```

## 3. Quick test in terminal
```bash
python run_model.py --input "[5.1,3.5,1.4,0.2]"    # → setosa
python run_model.py --input "[6.0,2.9,4.5,1.5]"    # → versicolor
python run_model.py --input "[6.7,3.0,5.2,2.3]"    # → virginica
```

## 4. Run the API  ← *ML engineer part*
**Terminal 1:**
```bash
python app.py                            # API on http://127.0.0.1:5001
```
**Terminal 2:**
```bash
curl localhost:5001/health
# {"status":"ok"}

curl -X POST localhost:5001/predict -H "Content-Type: application/json" -d '{"features":[6.0,2.9,4.5,1.5]}'
# {"flower":"versicolor","prediction":1}
```
Stop the API: **Ctrl+C** in Terminal 1 (port 5001 must be free for Docker).

## 5. Run in Docker  ← *MLOps part: package once, run anywhere*
**Terminal 1:**
```bash
docker build -t iris-api .               # build image from Dockerfile
docker images | grep iris-api            # see the image
docker run -p 5001:5001 iris-api         # run container (IP shows 172.17.x.x)
```
**Terminal 2:** same curl commands as step 4, same answers, now from the container.
```bash
docker ps                                # see the running container
```
Stop: **Ctrl+C** in Terminal 1.

## 6. CI/CD with GitHub Actions  ← *MLOps part: automation*
```bash
git add .
git commit -m "Show flower names in output"
git push origin main
```
Then on GitHub → **Actions** tab → *"CI - Train and Save the model"*
- Runs on Python **3.11** and **3.12**
- Installs deps → runs `train.py` → uploads `artifacts/` (model.pkl + metrics.json)

---

## If something breaks
| Error | Fix |
|---|---|
| `permission denied ... docker.sock` | Log out + log in, or run `newgrp docker` |
| `port is already allocated` / address in use | Stop the other app with Ctrl+C (only one can use 5001) |
| `docker: command not found` | `sudo apt install -y docker.io` |
| `ModuleNotFoundError: flask` | `source .venv/bin/activate` then `pip install -r requirements.txt` |
| `Model file not found` | Run `python train.py` first |
| `DEPRECATED: legacy builder` | Only a warning, ignore (or `sudo apt install docker-buildx`) |

## Who does what (say this while demoing)
- **Data scientist:** `train.py` → accurate model (`model.pkl`)
- **ML engineer:** `app.py` → API so apps and websites can use the model
- **MLOps engineer:** `Dockerfile` + `ci.yml` → automate, package and ship faster and safer
