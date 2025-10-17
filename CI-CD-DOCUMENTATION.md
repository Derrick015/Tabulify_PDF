# CI/CD Pipeline Documentation

## Overview

This document describes the GitLab CI/CD pipeline for the AI Powered PDF Table Extractor project. The pipeline includes comprehensive testing, security scanning, building, and deployment stages.

## Pipeline Stages

The pipeline consists of 4 main stages that run sequentially:

```
test → security → build → deploy
```

### 1. Test Stage

**Job: `test`**

- **Purpose**: Runs all unit and integration tests with coverage reporting
- **Image**: `python:3.11-slim`
- **Key Actions**:
  - Installs the package with development dependencies
  - Runs pytest with verbose output and coverage reporting
  - Generates HTML and terminal coverage reports
- **Artifacts**: 
  - Coverage reports (HTML format)
  - Coverage data files
- **Triggers**: Runs on `main`, `master` branches, and merge requests
- **Failure Policy**: Pipeline **fails** if tests fail (`allow_failure: false`)

**Coverage Badge**: The pipeline extracts the test coverage percentage and displays it as a GitLab badge.

### 2. Security Stage

The security stage runs three parallel jobs to scan for different types of vulnerabilities:

#### 2.1 Dependency Scanning (`security:dependency-scan`)

- **Purpose**: Scans Python dependencies for known vulnerabilities
- **Tools Used**:
  - `pip-audit`: Official PyPA tool for vulnerability scanning
  - `safety`: Additional security checker for Python dependencies
- **Triggers**: Runs on `main`, `master`, and merge requests
- **Failure Policy**: Reports issues but doesn't block pipeline (`allow_failure: true`)

#### 2.2 Code Security Scanning (`security:code-scan`)

- **Purpose**: Analyzes Python source code for security issues
- **Tool Used**: `bandit` - SAST (Static Application Security Testing) tool
- **Scan Targets**: `src/` directory and `app.py`
- **Severity Levels**: Reports medium and high severity issues
- **Artifacts**: 
  - `bandit-report.json` - Detailed security report
  - GitLab SAST report for security dashboard
- **Configuration**: Uses `.bandit` configuration file
- **Triggers**: Runs on `main`, `master`, and merge requests
- **Failure Policy**: Reports issues but doesn't block pipeline (`allow_failure: true`)

#### 2.3 Secret Scanning (`security:secret-scan`)

- **Purpose**: Scans codebase for exposed secrets, API keys, and credentials
- **Tool Used**: `gitleaks` v8.18.1
- **Scan Coverage**: Entire repository
- **Configuration**: Uses `.gitleaksignore` to exclude false positives
- **Triggers**: Runs on `main`, `master`, and merge requests
- **Failure Policy**: Reports issues but doesn't block pipeline (`allow_failure: true`)

### 3. Build Stage

**Job: `build_docker_image`**

- **Purpose**: Builds Docker image and pushes to Google Artifact Registry
- **Image**: `docker:latest` with `docker:dind` service
- **Key Actions**:
  - Authenticates with Google Cloud using service account
  - Builds Docker image with two tags:
    - `$CI_COMMIT_SHORT_SHA` - Unique commit identifier
    - `latest` - Latest stable version
  - Pushes both tags to Google Artifact Registry
- **Dependencies**: Waits for `test` job to pass
- **Triggers**: Only runs on `main` and `master` branches
- **Registry**: `us-central1-docker.pkg.dev`

### 4. Deploy Stage

The deploy stage includes container security scanning followed by deployment:

#### 4.1 Container Scanning (`security:container-scan`)

- **Purpose**: Scans the built Docker image for vulnerabilities
- **Tool Used**: `trivy` v0.48.3 by Aqua Security
- **Scan Focus**: HIGH and CRITICAL severity vulnerabilities
- **Artifacts**: 
  - `trivy-report.json` - Detailed vulnerability report
  - GitLab container scanning report
- **Dependencies**: Waits for `build_docker_image` to complete
- **Triggers**: Only runs on `main` and `master` branches
- **Failure Policy**: Reports issues but doesn't block deployment (`allow_failure: true`)

#### 4.2 GKE Deployment (`deploy_to_gke`)

- **Purpose**: Deploys the Docker image to Google Kubernetes Engine
- **Image**: `google/cloud-sdk:latest`
- **Key Actions**:
  - Authenticates with Google Cloud
  - Gets GKE cluster credentials
  - Updates deployment with new image tag
  - Verifies deployment rollout status
- **Dependencies**: 
  - Waits for `build_docker_image`
  - Waits for `security:container-scan`
- **Triggers**: Only runs on `main` and `master` branches
- **Condition**: Only deploys if all previous stages succeed (`when: on_success`)

## Required GitLab CI/CD Variables

The following environment variables must be configured in your GitLab project settings:

| Variable | Description | Example |
|----------|-------------|---------|
| `GCP_SERVICE_KEY` | Base64-encoded GCP service account JSON key | `<base64-encoded-json>` |
| `PROJECT_ID` | GCP Project ID (already set in pipeline) | `deron-innovations` |

### Setting up GCP_SERVICE_KEY

1. Create a service account in Google Cloud Console
2. Grant necessary permissions:
   - `Artifact Registry Writer`
   - `Kubernetes Engine Developer`
   - `Service Account User`
3. Create and download the JSON key
4. Base64 encode the key:
   ```bash
   cat key.json | base64 -w 0 > encoded-key.txt
   ```
5. Add the encoded key to GitLab:
   - Go to: Settings → CI/CD → Variables
   - Add variable: `GCP_SERVICE_KEY`
   - Type: Variable
   - Protected: ✓ (recommended)
   - Masked: ✓ (recommended)

## Security Configuration Files

### `.bandit`
Configures the bandit security scanner:
- Excludes test directories and virtual environments
- Sets minimum severity level to MEDIUM
- Sets minimum confidence level to MEDIUM

### `.gitleaksignore`
Allows you to exclude false positives from secret scanning:
- Add fingerprints of known false positives
- Exclude specific files or patterns if needed

## Viewing Security Reports

### GitLab Security Dashboard
If you have GitLab Ultimate, security reports will appear in:
- Project → Security & Compliance → Security Dashboard
- Merge Request → Security tab

### Artifact Reports
All security scan reports are saved as artifacts and can be downloaded:
- `bandit-report.json` - Code security issues
- `trivy-report.json` - Container vulnerabilities

Reports are retained for 1 week.

## Pipeline Behavior

### Merge Requests
- ✅ Tests run
- ✅ All security scans run (dependency, code, secrets)
- ❌ Build does NOT run
- ❌ Deploy does NOT run

### Main/Master Branch
- ✅ Tests run (must pass)
- ✅ All security scans run (informational)
- ✅ Build runs (only if tests pass)
- ✅ Container scan runs (only if build succeeds)
- ✅ Deploy runs (only if all previous stages succeed)

## Troubleshooting

### Test Stage Fails
1. Check test output in the job logs
2. Run tests locally: `pytest -v`
3. Ensure all dependencies are in `pyproject.toml`

### Security Scans Report Issues
Security scans are informational and don't block the pipeline. Review the reports:
1. Download the artifact JSON files
2. Address high-priority vulnerabilities
3. Update dependencies: `pip install --upgrade <package>`

### Build Fails
1. Verify `GCP_SERVICE_KEY` is correctly configured
2. Check Docker build logs
3. Ensure Dockerfile is correct
4. Verify GCP service account has Artifact Registry permissions

### Deploy Fails
1. Verify GKE cluster is accessible
2. Check service account has Kubernetes Engine permissions
3. Verify `kubernetes-deployment.yaml` is correct
4. Check deployment name matches in script

## Best Practices

1. **Always create merge requests** - This runs tests and security scans before merging
2. **Review security reports** - Even though they don't block, address critical issues
3. **Keep dependencies updated** - Regular updates reduce vulnerabilities
4. **Monitor coverage** - Aim to maintain or improve test coverage
5. **Use semantic versioning** - Consider tagging releases with version numbers

## Performance Optimization

The pipeline is optimized for speed:
- Security scans run in parallel
- Build only runs after tests pass
- Container scan runs in parallel with other deploy stage jobs
- Dependencies are cached where possible

## Future Enhancements

Consider adding:
- Performance testing stage
- Integration tests with external services
- Automated rollback on deployment failure
- Multi-environment deployments (dev, staging, prod)
- Slack/email notifications for pipeline status

