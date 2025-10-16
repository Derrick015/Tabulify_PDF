# GitLab CI/CD Setup Guide

This guide will help you set up automated CI/CD for your PDF Table Extractor app.

## 📋 Prerequisites

- GitLab account with your repository
- GCP project with Artifact Registry and GKE cluster
- GCP service account with appropriate permissions

---

## 🔧 Step 1: Create GCP Service Account

1. **Go to GCP Console** → IAM & Admin → Service Accounts

2. **Create a new service account:**
   ```bash
   gcloud iam service-accounts create gitlab-ci-cd \
     --display-name="GitLab CI/CD Service Account"
   ```

3. **Grant necessary roles:**
   ```bash
   # Artifact Registry Writer (to push Docker images)
   gcloud projects add-iam-policy-binding deron-innovations \
     --member="serviceAccount:gitlab-ci-cd@deron-innovations.iam.gserviceaccount.com" \
     --role="roles/artifactregistry.writer"
   
   # Kubernetes Engine Developer (to deploy to GKE)
   gcloud projects add-iam-policy-binding deron-innovations \
     --member="serviceAccount:gitlab-ci-cd@deron-innovations.iam.gserviceaccount.com" \
     --role="roles/container.developer"
   
   # Service Account User (required for GKE)
   gcloud projects add-iam-policy-binding deron-innovations \
     --member="serviceAccount:gitlab-ci-cd@deron-innovations.iam.gserviceaccount.com" \
     --role="roles/iam.serviceAccountUser"
   ```

4. **Create and download the key:**
   ```bash
   gcloud iam service-accounts keys create gitlab-ci-key.json \
     --iam-account=gitlab-ci-cd@deron-innovations.iam.gserviceaccount.com
   ```

5. **Encode the key to base64:**
   
   **On Linux/Mac:**
   ```bash
   cat gitlab-ci-key.json | base64 -w 0 > gitlab-ci-key-base64.txt
   ```
   
   **On Windows (PowerShell):**
   ```powershell
   [Convert]::ToBase64String([IO.File]::ReadAllBytes("gitlab-ci-key.json")) | Out-File gitlab-ci-key-base64.txt
   ```

---

## 🔐 Step 2: Configure GitLab CI/CD Variables

1. **Go to your GitLab repository** → Settings → CI/CD → Variables

2. **Add the following variables:**

   | Key | Value | Protected | Masked |
   |-----|-------|-----------|---------|
   | `GCP_SERVICE_KEY` | Contents of `gitlab-ci-key-base64.txt` | ✅ | ✅ |
   | `OPENAI_API_KEY` | Your OpenAI API key | ✅ | ✅ |

   **How to add:**
   - Click "Add variable"
   - Enter the key name
   - Paste the value
   - Check "Protected" (only runs on protected branches)
   - Check "Masked" (hides value in logs)
   - Click "Add variable"

---

## ⚙️ Step 3: Update `.gitlab-ci.yml`

Edit `.gitlab-ci.yml` and update these values:

```yaml
variables:
  GKE_CLUSTER_NAME: your-actual-cluster-name  # ← Update this
  GKE_ZONE: us-central1-a                     # ← Update this (or region for regional cluster)
```

**To find your cluster name and zone:**
```bash
gcloud container clusters list
```

---

## 🏃 Step 4: Enable GitLab Runner

**Option 1: Use GitLab.com Shared Runners (Recommended)**
- Go to Settings → CI/CD → Runners
- Enable "Shared runners"
- No additional setup needed!

**Option 2: Use Your Own Runner**
- Follow [GitLab Runner installation guide](https://docs.gitlab.com/runner/install/)
- Register runner with your project

---

## 🚀 Step 5: Deploy!

1. **Commit and push your changes:**
   ```bash
   git add .gitlab-ci.yml kubernetes-deployment.yaml
   git commit -m "Add GitLab CI/CD pipeline"
   git push origin main
   ```

2. **Watch the pipeline:**
   - Go to your GitLab repository → CI/CD → Pipelines
   - Click on the running pipeline to see progress
   - Monitor build and deploy stages

3. **Check deployment:**
   ```bash
   # View pods
   kubectl get pods
   
   # View service (get external IP)
   kubectl get svc ai-powered-pdf-table-extractor-service
   
   # View logs
   kubectl logs -f deployment/ai-powered-pdf-table-extractor
   ```

---

## 📊 Pipeline Stages

### **Stage 1: Build**
- Builds Docker image
- Tags with commit SHA and `latest`
- Pushes to GCP Artifact Registry

### **Stage 2: Deploy**
- Creates/updates Kubernetes secret with OpenAI API key
- Applies Kubernetes deployment
- Updates image to specific commit SHA
- Waits for rollout to complete
- Shows service information

---

## 🔄 How It Works

**On every push to `main` or `master` branch:**

1. ✅ GitLab detects changes
2. ✅ Runs build stage → Docker image created and pushed
3. ✅ Runs deploy stage → Updates Kubernetes deployment
4. ✅ Rolling update → Zero downtime!
5. ✅ Your app is live with the latest code

---

## 🛠️ Useful Commands

**View pipeline status:**
```bash
# In GitLab UI: CI/CD → Pipelines
```

**Rollback to previous version:**
```bash
kubectl rollout undo deployment/ai-powered-pdf-table-extractor
```

**Check deployment history:**
```bash
kubectl rollout history deployment/ai-powered-pdf-table-extractor
```

**Manual trigger:**
- Go to CI/CD → Pipelines → "Run pipeline"

---

## 🐛 Troubleshooting

### **Build fails with authentication error:**
- Verify `GCP_SERVICE_KEY` variable is set correctly
- Ensure service account has `artifactregistry.writer` role

### **Deploy fails with kubectl error:**
- Check service account has `container.developer` role
- Verify cluster name and zone are correct
- Ensure cluster is in the same project

### **Pipeline doesn't run:**
- Check if branch is `main` or `master` (only these trigger pipeline)
- Verify GitLab Runner is enabled
- Check `.gitlab-ci.yml` syntax

### **Secret creation fails:**
- Verify `OPENAI_API_KEY` variable is set in GitLab

---

## 🎯 Next Steps

**Add more stages:**
- Testing stage (run unit tests before build)
- Staging environment (deploy to staging before production)
- Manual approval (require approval before production deploy)

**Example with testing:**
```yaml
stages:
  - test
  - build
  - deploy

test:
  stage: test
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - pytest tests/
  only:
    - main
    - master
```

---

## 📚 Resources

- [GitLab CI/CD Documentation](https://docs.gitlab.com/ee/ci/)
- [GCP Service Accounts](https://cloud.google.com/iam/docs/service-accounts)
- [Kubernetes Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)

---

**Questions?** Check the troubleshooting section or GitLab pipeline logs for detailed error messages.

