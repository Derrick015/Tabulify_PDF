
## Deployment with Google Kubernetes Engine (GKE)

1. Create your repo and connect to gitlab. You can mirror to github if you like. 
2. Create a pyproject.toml file
3. Create a Dockerfile, check previous work.
4. EXPOSE 8501 if using streamlit as it is the default
5. Create a .dockerignore with similar bits in the .gitignore
6. Build image with "docker build -t pdf-table-extractor ." actually not in quotes
7. I used docker installed in ubuntu (wsl) for the above. docker desktop could also be used
6. Create and run a container with docker run --rm -p 8501:8501 -e OPENAI_API_KEY=sk-your-actual-key-here pdf-table-extractor
7. Check it out at local host http://localhost:8501/ if i used 8501 to ensure it works
8. Create a kubernetes-deployment.yaml file. Ensure correct project name and app name is used
9. Enable the following: 

    Kubernetes Engine API
    Container Registry API
    Compute Engine API
    Cloud Build API
    Cloud Storage API
    IAM API
10. Go to kubernetes engine in gcp, clusters and then create
11. Set name and region which should match the details in your kubernetes-deployment file
12. Go to networking and enable access using DNS and leave the access using IPv4 addresses enabled. 
13. Click create
14. Go to artifacts registry and create one. Select docker and set the region as the kubernetes region and leave all else the same
15. Go to service account and create a new one.
16. In permissions go for
    Storage Object Admin
    Storage Object Viewer
    Owner
    Artifact Registry Admin
    Artifact Registry Writer
17. Download the Json key. Do this by clicking on the three donts and going to manage keys.
18. Convert the json key to based 64. You can use the git bash terminal for this. run this in there and copy the key: cat yourkey.json | base64 -w 0. Note do not add in the =(whatever..) bit.
19. Go to your gitlab project. Settings, CI/CD, add variable and click add variable
20. Add the base64 key inthe value section and the key shoud be what ever key name you used below in the gitlab-ci.yml file. 
    mine was GCP_SERVICE_KEY
''''
    # Deploy to GKE
deploy:
  stage: deploy
  image: google/cloud-sdk:alpine
  before_script:
    # Authenticate with GCP
    - echo $GCP_SERVICE_KEY | base64 -d > ${HOME}/gcp-key.json
    - gcloud auth activate-service-account --key-file ${HOME}/gcp-key.json
    - gcloud config set project $GCP_PROJECT_ID
''''
21. Create another variable for the OPEN_API_KEY
22. Got to your kubernetes cluster, click connect and run in cloud shell and hit enter to run the pre-populated line.
23. add secret key in your.env and hit enter in the cloud shell. In my case it was OPEN_AI_KEY:

kubectl create secret generic llmops-secrets \
--from-literal=OPEN_AI_KEY="your_actual_key"


