# Custom Domain Setup for GKE Deployment

## Prerequisites
- A custom domain that you own
- Access to your domain's DNS settings
- GKE cluster running with your application

## Step 1: Reserve a Static IP Address

Run this in Cloud Shell or your terminal with gcloud CLI:

```bash
# Reserve a global static IP address
gcloud compute addresses create pdf-extractor-ip \
  --global \
  --project=deron-innovations

# Get the IP address (save this for DNS configuration)
gcloud compute addresses describe pdf-extractor-ip --global --project=deron-innovations
```

**Important:** Copy the IP address shown - you'll need it for DNS configuration.

## Step 2: Apply Kubernetes Ingress Configuration

Apply the updated Kubernetes configuration:

```bash
# Make sure you're connected to your cluster
gcloud container clusters get-credentials [YOUR-CLUSTER-NAME] \
  --region=[YOUR-REGION] \
  --project=deron-innovations

# Apply the managed certificate
kubectl apply -f kubernetes-ingress.yaml

# Apply the ingress
kubectl apply -f kubernetes-deployment.yaml
```

## Step 3: Configure DNS Records

In your domain registrar's DNS settings (e.g., GoDaddy, Namecheap, Cloudflare, Google Domains):

1. Create an **A Record**:
   - **Name/Host:** `@` (for root domain) or `www` (for subdomain)
   - **Type:** A
   - **Value/Points to:** [The static IP from Step 1]
   - **TTL:** 3600 (or default)

2. (Optional) If you want both `yourdomain.com` and `www.yourdomain.com`:
   - Create another A Record for the other prefix
   - OR create a CNAME record pointing `www` to `yourdomain.com`

Example:
```
Type    Name    Value               TTL
A       @       34.120.123.45       3600
A       www     34.120.123.45       3600
```

## Step 4: Wait for Certificate Provisioning

It takes **10-60 minutes** for Google to provision the SSL certificate. Check status:

```bash
# Check managed certificate status
kubectl describe managedcertificate pdf-extractor-cert

# Check ingress status
kubectl describe ingress pdf-extractor-ingress

# Wait for certificate to show "Active"
kubectl get managedcertificate pdf-extractor-cert -w
```

## Step 5: Verify the Setup

1. Wait for DNS propagation (can take up to 48 hours, usually much faster)
2. Check DNS propagation: https://dnschecker.org
3. Access your app at: `https://yourdomain.com`

## Troubleshooting

### Certificate stuck in "Provisioning"
- Verify DNS is pointing to the correct IP
- Ensure the domain in `managedcertificate` exactly matches your DNS record
- Check: `kubectl describe managedcertificate pdf-extractor-cert`

### 502 Bad Gateway
- Check if pods are running: `kubectl get pods`
- Check service: `kubectl get svc`
- Check logs: `kubectl logs -l app=ai-powered-pdf-table-extractor`

### 404 Not Found
- Verify the Ingress is correctly routing: `kubectl describe ingress pdf-extractor-ingress`
- Check backend service health: `kubectl get svc ai-powered-pdf-table-extractor-service`

### DNS not resolving
- Use `nslookup yourdomain.com` to verify DNS
- Check DNS propagation status
- Verify A record is pointing to the static IP

## Additional Configuration

### Multiple Domains
To support multiple domains (e.g., both `example.com` and `www.example.com`), update the `managedcertificate` resource:

```yaml
spec:
  domains:
    - yourdomain.com
    - www.yourdomain.com
```

### Custom SSL Certificate (Alternative)
If you prefer to use your own SSL certificate instead of Google-managed:

```bash
# Create a secret with your certificate
kubectl create secret tls pdf-extractor-tls \
  --cert=path/to/cert.pem \
  --key=path/to/key.pem

# Update ingress to use the secret (see kubernetes-ingress.yaml for details)
```

## Monitoring

Check your application status:
```bash
# Get all resources
kubectl get all

# Check ingress IP
kubectl get ingress pdf-extractor-ingress

# View logs
kubectl logs -f -l app=ai-powered-pdf-table-extractor
```

## Cost Considerations
- Static IP: ~$0.01/hour when not in use, free when in use
- Ingress/Load Balancer: ~$0.025/hour + data processing charges
- Certificate: Free with Google-managed certificates

