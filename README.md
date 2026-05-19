# SageMaker Wine Quality Demo

You will train three classifiers (Logistic Regression, KNN, XGBoost) on the UCI Wine Quality dataset, compare their performance, deploy the winning model as a SageMaker real-time endpoint, and call that endpoint from a Streamlit web app. The Streamlit app can run either on your laptop or on an EC2 instance — both options are documented below.

## Architecture

```
[Browser]
   |
   v
[Streamlit app]               <- runs on your laptop OR on EC2
   |
   |  boto3.invoke_endpoint()
   v
[SageMaker endpoint]          <- runs on AWS
   |
   v
[S3: model.tar.gz]
```

The SageMaker endpoint is not publicly reachable. Every call requires an AWS-signed request (SigV4). Authentication uses your Learner Lab credentials when running locally, or the EC2 instance profile when running on EC2.

## Repo layout

```
.
├── README.md
├── pipeline.py              # entry point: train all 3 models, pick winner, package
├── deploy_endpoint.py       # creates the SageMaker endpoint
├── streamlit_app.py         # the UI
├── user-data.sh             # EC2 bootstrap script
└── src/
    ├── data.py              # dataset loading, splitting, label binning
    ├── models.py            # pipeline factory: 3 candidate models
    ├── evaluate.py          # metrics and comparison table
    ├── inference.py         # SageMaker container entry point
    └── requirements.txt     # extra packages for the SageMaker container
```

### What each file does

`pipeline.py` is the orchestrator. It loads data, builds three model pipelines, fits each, evaluates them, prints a comparison table, picks the winner by test accuracy, and saves it as `model.tar.gz`.

`src/data.py` downloads the dataset and splits it. The label binning function collapses 6 quality scores into 3 classes (low/medium/high) to keep class imbalance manageable for class.

`src/models.py` returns a dict of named sklearn `Pipeline` objects — one per candidate model. Each pipeline is `StandardScaler -> classifier`. Standardization is essential for distance-based models (KNN) and scale-sensitive linear models (Logistic Regression); it's harmless for tree models (XGBoost). Using the same pipeline shape for all three lets training, evaluation, and inference treat them interchangeably.

`src/evaluate.py` computes train/test accuracy and macro F1 for any pipeline, prints a comparison table, and selects the winner.

`src/inference.py` is the SageMaker container entry point. It's generic — it loads any pickled sklearn `Pipeline`, regardless of which classifier is inside. Switching to a different model in `models.py` requires no changes here.

## Learner Lab constraints

Read these once. They explain why later steps look the way they do.

- Region must be `us-east-1`.
- The IAM role you must use is `LabRole`. You cannot create custom roles.
- SageMaker endpoint instances are limited to medium / large / xlarge sizes. We use `ml.m5.large`.
- AWS credentials issued by Learner Lab are temporary. They expire when the lab session ends and need to be refreshed.
- Endpoints bill per second. Delete them before you end the lab.

---

## Part 1 — Train and compare models

You will run `pipeline.py` inside a SageMaker Notebook Instance. The notebook environment already has Python, pip, and AWS credentials configured.

### 1.1 Open a SageMaker Notebook Instance

In the AWS Console, open SageMaker → Notebook → Notebook instances → Create notebook instance.

- Notebook instance name: anything, e.g. `wine-demo-notebook`
- Instance type: `ml.t3.medium`
- IAM role: `LabRole`
- Leave other settings at defaults. Click Create.

Wait until the status shows `InService` (about 3 minutes), then click `Open JupyterLab`.

### 1.2 Upload the project files

In JupyterLab, drag `pipeline.py`, `deploy_endpoint.py`, and the `src/` folder into the file browser on the left. Open a new terminal (File → New → Terminal).

### 1.3 Run the pipeline

```bash
pip install xgboost==1.7.6 scikit-learn==1.2.2 pandas joblib
python pipeline.py
```

You will see output like:

```
Loading dataset...
Dataset shape: (1599, 12)
Train: (1279, 11), Test: (320, 11)

Training logistic_regression...
Training knn...
Training xgboost...

Model                   Train Acc   Test Acc    Test F1
------------------------------------------------------
logistic_regression        0.5934     0.5781     0.5654
knn                        0.6552     0.5500     0.5421
xgboost                    0.9984     0.6781     0.6612

Winner: xgboost

Detailed report for xgboost:
              precision    recall  f1-score   support
         low       0.74      0.78      0.76       128
      medium       0.61      0.59      0.60       128
        high       0.70      0.66      0.68        64
    accuracy                           0.68       320
   ...

Saved: model_artifact/model.joblib
Packaged: model_artifact/model.tar.gz
```

The exact numbers will vary depending on the dataset version. The winning model becomes `model_artifact/model.tar.gz`.

### 1.4 Read the comparison

A few things worth noticing in your output:

- **XGBoost typically overfits** on small tabular datasets — the gap between train and test accuracy is large. This is normal and addressable with regularization (lower `max_depth`, fewer estimators).
- **KNN is sensitive to scale** — without `StandardScaler`, it would perform much worse. The scaler in the pipeline is doing real work.
- **Logistic Regression is the baseline.** If a more complex model can't beat it by much, the additional complexity isn't earning its keep.

The comparison framework is the actual lesson: by giving every model the same pipeline shape, you can swap algorithms without changing your evaluation, deployment, or serving code.

---

## Part 2 — Create an S3 bucket and upload the model

SageMaker reads the model artifact from S3, not from the notebook's local disk. So you need an S3 bucket.

### 2.1 Create the bucket

In the terminal:

```bash
BUCKET=wine-demo-<your-name>-<4-random-digits>
aws s3 mb s3://$BUCKET --region us-east-1
```

Bucket names must be globally unique across all of AWS. The random suffix is not optional. If you see `BucketAlreadyExists`, pick a different suffix.

### 2.2 Upload the model

```bash
aws s3 cp model_artifact/model.tar.gz s3://$BUCKET/wine/model.tar.gz
aws s3 ls s3://$BUCKET/wine/
```

The second command should list the file you just uploaded.

---

## Part 3 — Deploy the SageMaker endpoint

### 3.1 Edit deploy_endpoint.py

Open `deploy_endpoint.py` in JupyterLab and edit the three variables at the top:

```python
BUCKET = "wine-demo-yourname-1234"   # the bucket you just created
MODEL_S3_KEY = "wine/model.tar.gz"
ENDPOINT_NAME = "wine-endpoint"
```

### 3.2 Run it

In the terminal:

```bash
python deploy_endpoint.py
```

Endpoint creation takes 5 to 8 minutes. While you wait, open `src/inference.py` and read it. The four functions (`model_fn`, `input_fn`, `predict_fn`, `output_fn`) are the contract between your code and SageMaker's model server. Understanding this contract is the main concept of the session.

When the script finishes, you will see a smoke test response printed:

```
{"probabilities": [[0.85, 0.13, 0.02]], "predictions": [0], "labels": ["low"]}
```

The endpoint is now live. You can verify in the SageMaker console under Inference → Endpoints.

### 3.3 About the model file format

The model is saved as a **joblib** file, which is a pickle variant optimized for sklearn objects. A few things worth knowing:

- **joblib preserves the entire Python object.** The full pipeline (scaler + classifier + fitted state) is in one file. Loading it gives you back exactly what you trained.
- **It's Python-only.** A joblib file can only be loaded by Python with the same library versions installed. You cannot load it in JavaScript, C++, or another language.
- **It's not safe for untrusted sources.** Loading a joblib file can execute arbitrary code. Never load one from someone you don't trust.
- **Other formats exist.** ONNX is portable across languages and runtimes. Framework-native formats (XGBoost's `.ubj`, TensorFlow's SavedModel) are more stable across versions but framework-specific. For sklearn pipelines specifically, joblib is the standard.

The SageMaker container needs every library that was in scope at training time to load the pickled pipeline. That's why `requirements.txt` includes `xgboost` even though the winning model might be Logistic Regression — the unpickler needs `xgboost` to be importable to reconstruct the dict of pipelines, even if it's not used in the final model. This is a real production gotcha: your inference dependencies must match your training environment, not just the final model's dependencies.

---

## Part 4 — Run the Streamlit app

You have two options. Choose one. Both call the same SageMaker endpoint and produce identical results.

### Option A — Run Streamlit on your laptop

This is the faster option. Recommended if you just want to see the system working.

#### A.1 Install Streamlit and boto3

On your laptop:

```bash
pip install streamlit boto3
```

#### A.2 Configure AWS credentials

In the Learner Lab, click `AWS Details` → `Show` next to AWS CLI. You will see three lines:

```
aws_access_key_id=ASIA...
aws_secret_access_key=...
aws_session_token=...
```

Create the file `~/.aws/credentials` (Linux/macOS) or `%USERPROFILE%\.aws\credentials` (Windows) with this content:

```ini
[default]
aws_access_key_id = ASIA...
aws_secret_access_key = ...
aws_session_token = ...
```

Create `~/.aws/config` with:

```ini
[default]
region = us-east-1
output = json
```

These credentials are temporary. When your lab session ends, they expire. You will need to copy fresh values from `AWS Details` and overwrite the file.

#### A.3 Verify credentials work

```bash
aws sts get-caller-identity
```

You should see output containing `assumed-role/voclabs/...`. If you see `Unable to locate credentials` or `ExpiredToken`, fix that first — Streamlit will not work until this command does.

#### A.4 Run the app

```bash
export ENDPOINT_NAME=wine-endpoint
export AWS_REGION=us-east-1
streamlit run streamlit_app.py
```

On Windows PowerShell:

```powershell
$env:ENDPOINT_NAME = "wine-endpoint"
$env:AWS_REGION = "us-east-1"
streamlit run streamlit_app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). Click `Predict`. You should see a class label and probability bars.

### Option B — Run Streamlit on EC2

This is the more realistic deployment. The EC2 instance authenticates to AWS via an IAM role, so there are no credentials in your code or config files.

#### B.1 Push the project to a public GitHub repo

Create a new public GitHub repo and push the project files. The EC2 user-data script clones it during boot.

If you already have a parent repo (e.g. a course repo) with this project as a subfolder, that works too — see the `SUBFOLDER` variable below.

#### B.2 Edit user-data.sh

Open `user-data.sh` and edit the variables at the top:

```bash
GIT_REPO="https://github.com/yourname/wine-demo.git"
SUBFOLDER=""                          # leave empty if streamlit_app.py is at repo root
APP_FILE="streamlit_app.py"
ENDPOINT_NAME="wine-endpoint"
```

`SUBFOLDER` is the path inside the repo to the directory containing `streamlit_app.py`. If your repo layout is:

```
my-repo/
└── streamlit_app.py
```

then `SUBFOLDER=""`.

If your layout is:

```
my-repo/
└── session10/
    └── wine-demo/
        └── streamlit_app.py
```

then `SUBFOLDER="session10/wine-demo"`. No leading or trailing slashes.

#### B.3 Launch the EC2 instance

In the AWS Console: EC2 → Launch instances.

- Name: `wine-demo-app`
- AMI: Amazon Linux 2023
- Instance type: `t3.micro`
- Key pair: Proceed without a key pair (you will use EC2 Instance Connect)
- Network settings → Edit:
  - Allow SSH traffic from: My IP
  - Add security group rule: Type `Custom TCP`, Port `8501`, Source `0.0.0.0/0`
- Advanced details:
  - IAM instance profile: `LabInstanceProfile`
  - User data: paste the entire contents of `user-data.sh`

Click Launch.

#### B.4 Wait and verify

The user-data script runs during the first boot. It takes about 2-3 minutes. After waiting, copy the instance's public IPv4 address from the EC2 console.

In your laptop's terminal:

```bash
curl -v http://<public-ip>:8501
```

You should see HTTP response headers and HTML. If you see `Connection refused`, the script is still running or has failed (see Troubleshooting below). If you see `Connection timed out`, check the security group inbound rules.

Open `http://<public-ip>:8501` in your browser. The Streamlit app loads. Click Predict.

#### B.5 Useful EC2 commands

To SSH into the instance: EC2 console → select instance → Connect → EC2 Instance Connect → Connect. This opens a browser-based terminal.

```bash
# Was user-data successful?
ls /opt/wine-app/.userdata-success

# Service status
sudo systemctl status streamlit

# Live logs
sudo journalctl -u streamlit -f

# Restart after editing the app
sudo systemctl restart streamlit
```

---

## Part 5 — Teardown

You must do this. Endpoints and EC2 instances bill while running, even if you are not using them.

In the SageMaker terminal:

```python
import boto3
sm = boto3.client("sagemaker", region_name="us-east-1")
sm.delete_endpoint(EndpointName="wine-endpoint")
sm.delete_endpoint_config(EndpointConfigName="wine-endpoint")
```

Or via console: SageMaker → Inference → Endpoints → select → Delete. Then Endpoint configurations → Delete. Then Models → Delete.

In the EC2 console: select your instance → Instance state → Terminate instance.

In the SageMaker console: Notebook instances → select → Stop, then Delete.

In the S3 console (or via CLI):

```bash
aws s3 rm s3://$BUCKET --recursive
aws s3 rb s3://$BUCKET
```

Finally, end the Learner Lab session.

---

## Troubleshooting

### `aws sts get-caller-identity` fails

| Error | Fix |
|---|---|
| `Unable to locate credentials` | `~/.aws/credentials` is missing or in the wrong location |
| `InvalidClientTokenId` | You copied only two of the three keys — the session token is missing |
| `ExpiredToken` | Lab session ended. Restart the lab and copy fresh credentials |

### SageMaker endpoint deployment fails

| Error | Fix |
|---|---|
| `iam:PassRole` denied | You are using a role other than LabRole. Use only LabRole |
| `ResourceLimitExceeded` | Wrong instance type. Use `ml.m5.large` |
| Endpoint stuck in `Creating` for >15 min | Check CloudWatch logs for the endpoint. Usually a problem in `inference.py` |
| `ModuleNotFoundError` in CloudWatch | A library used at training time is missing from `src/requirements.txt` |

### Streamlit on EC2 — connection refused

The user-data script failed before starting Streamlit. SSH in and check:

```bash
sudo cat /var/log/cloud-init-output.log | tail -80
```

Look for the last error. Common causes: typo in `GIT_REPO`, repo is private, wrong `SUBFOLDER`, wrong `APP_FILE`.

### Streamlit on EC2 — connects but page won't load

Almost always a WebSocket issue. The `user-data.sh` already disables CORS and XSRF protection to handle this. If it still fails, check `sudo journalctl -u streamlit -n 50` for Python errors in `streamlit_app.py`.

### Streamlit locally — `NoCredentialsError`

`~/.aws/credentials` is missing, malformed, or in the wrong location. Run `aws sts get-caller-identity` to confirm credentials are reachable, then check that `streamlit run` was started in a terminal that can see the same credentials.

### Streamlit anywhere — `ResourceNotFound: Endpoint ... not found`

Either the endpoint is in a different region than your boto3 client, or it has been deleted, or `ENDPOINT_NAME` doesn't match what you deployed.

---

## Sample requests

The endpoint accepts JSON or CSV.

JSON:
```json
{"instances": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]}
```

CSV:
```
7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4
```

Feature order: `fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol`.

Response:
```json
{
  "probabilities": [[0.85, 0.13, 0.02]],
  "predictions": [0],
  "labels": ["low"]
}
```

---

## Going further

If you want to extend this project after class:

- **Add a model.** Open `src/models.py`, add a new entry to the dict (e.g., `RandomForestClassifier`, `GradientBoostingClassifier`). Run `python pipeline.py`. Nothing else needs to change — the comparison, winner selection, and serving code all handle it automatically. This is the payoff of the pipeline abstraction.
- **Tune hyperparameters.** Wrap each model in `GridSearchCV` or `RandomizedSearchCV`. The pipeline shape stays the same.
- **Try ONNX.** Convert the winning sklearn pipeline to ONNX with `skl2onnx`, then deploy with the same SageMaker endpoint pattern. You'll need to change `inference.py` to use `onnxruntime` instead of `joblib`, but the contract stays identical.
- **Add proper authentication to the Streamlit app.** Right now anyone who knows the EC2 public IP can use it. In production you'd put it behind nginx with HTTP basic auth, or behind an Application Load Balancer with Cognito.
