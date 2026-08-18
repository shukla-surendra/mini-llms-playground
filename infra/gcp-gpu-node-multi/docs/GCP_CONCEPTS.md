# GCP Concepts, Coming From AWS

Companion to the [`gcp-gpu-node`](../README.md) module itself — that README covers
*what this module creates and how to run it*; this doc covers *the GCP concepts
underneath it*, for the same reason [`../../aws-gpu-node/docs/`](../../aws-gpu-node/docs/)
exists alongside that module. Every concept below is grounded in this module's actual
Terraform, not left abstract — look for the "In this repo" callouts.

## 1. The AWS → Azure → GCP mapping

| Concept | AWS | Azure | GCP |
|---|---|---|---|
| Top-level boundary | AWS Organization | Tenant | Organization |
| Environment grouping | AWS Account | Subscription | Project |
| Grouping inside top level | Organizational Unit (OU) | Management Group | Folder |
| Actual resources | EC2, S3, RDS | VMs, Storage | Compute Engine, GCS, GKE |
| Identity/access | IAM | Entra ID + RBAC | Cloud IAM |
| Machine identity | IAM Role | Managed Identity / Service Principal | Service Account |
| Long-lived machine credential | Access key | Client secret / certificate | Service account key |
| Temporary machine credential | IAM Role / STS | Managed Identity | Service account impersonation / Workload Identity Federation |

One caveat worth keeping in mind rather than smoothing over: **an AWS Account and a
GCP Project are not quite the same thing.** A GCP Project is simultaneously a resource
container, an IAM boundary, an API/service-enablement boundary, *and* commonly a
billing boundary, all at once — which is why GCP architecture ends up organized
around projects more heavily than the table above might suggest. AWS spreads those
same four roles across account + IAM + service-enablement-per-service + Billing more
loosely.

## 2. The resource hierarchy

```
Organization
│
├── Folder: Development
│   ├── Project: myapp-dev
│   └── Project: ml-dev
│
├── Folder: Production
│   ├── Project: myapp-prod
│   └── Project: ml-prod
│
└── Folder: Sandbox
    └── Project: experimentation
```

IAM policies can be attached at the Organization, Folder, Project, *or* individual
resource level, and **policies inherited from a higher level flow down** — grant a
role at the Folder level and everything underneath it inherits that grant.

**For a small personal setup, folders are optional.** A flat list of projects directly
under the Organization (or even no Organization at all, for a personal Google account)
is completely normal:

```
Organization
│
├── Project: gcp-learning
├── Project: llm-training
└── Project: experiments
```

**In this repo:** `gcp-gpu-node`'s `var.project_id` (`variables.tf`) targets exactly
one Project — this module doesn't touch Folders or the Organization level at all,
consistent with "you don't necessarily need folders for a small personal setup."

## 3. IAM: principal → role → permissions → resource

```
Principal (a user, group, or service account)
       ↓
    Role (a named bundle of permissions)
       ↓
  Permissions (individual allowed actions)
       ↓
   Resource (what the permissions apply to)
```

Concretely:

```
my-user@gmail.com
       ↓
roles/compute.admin
       ↓
Project: gcp-learning
       ↓
Can administer Compute Engine in that project
```

**In this repo:** `iam.tf` grants `roles/storage.objectAdmin` to this module's
service account, scoped to exactly the one GCS bucket this module creates —
`google_storage_bucket_iam_member` binds the role at the *resource* level (the
bucket), not the Project level, which is the tightest scope this hierarchy offers.
The self-stop permission goes one step further and binds a **custom** role (only
`compute.instances.get` + `compute.instances.stop`, nothing else) to the *specific
VM instance* resource — see that file's comment for why GCP's IAM Conditions don't
support the AWS-style "scope by tag" approach as directly, and why binding straight
to the one resource this module manages ends up just as tight in practice.

## 4. Service accounts: the machine identity

```
Application
     ↓
Service Account
     ↓
IAM roles
     ↓
GCP resources
```

A service account is GCP's direct analog of an AWS IAM Role used by an EC2 instance
— an identity a *workload* authenticates as, distinct from a human user identity.

```
Project: gcp-learning

Service Account:
    terraform-admin@my-project.iam.gserviceaccount.com

Roles:
    Compute Admin
    Storage Admin
```

**In this repo:** `iam.tf`'s `google_service_account.gpu`, attached to the VM in
`main.tf` via the `service_account { email = ... }` block. The VM authenticates as
this identity automatically through the metadata server — nothing is typed in or
stored as a file on the box.

## 5. Service account keys vs. the modern alternative

Historically: create a JSON key file, hand it to whatever needs to authenticate.

```
Terraform
   ↓
service-account.json
   ↓
Service Account
   ↓
GCP IAM
   ↓
Project resources
```

This is the direct analog of setting `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` —
and it has the exact same downside AWS long-lived access keys have: **possession of
the file is possession of everything the service account can do**, with no
expiration and no built-in revocation trigger. Google's own guidance is to avoid
minting these keys when an alternative exists.

**In this repo:** no service-account JSON key exists anywhere in this module. The VM
authenticates via its attached service account through the metadata server (rotating
credentials, revoked the instant the VM is destroyed) — the same "no long-lived
credential ever touches the box" property the sibling `aws-gpu-node` module gets from
an EC2 instance profile, via a different mechanism. You (the human operator) instead
authenticate locally with `gcloud auth application-default login`, which Terraform
picks up automatically — no key file for *you* either.

### Workload Identity Federation — the modern pattern for external workloads

For a workload running *outside* GCP entirely (GitHub Actions, a CI/CD pipeline, a
workload on AWS/Azure needing to reach GCP) that still needs to act as a GCP service
account, Workload Identity Federation lets it exchange a token from its *own*
identity provider for short-lived GCP credentials — no key file in a CI secrets
store at all:

```
GitHub Actions
      ↓
Workload Identity Federation
      ↓
GCP Service Account
      ↓
IAM
      ↓
GCP resources
```

Not used by this module (there's no CI/CD pipeline here, just a human running
`terraform apply` from a Mac) — worth knowing about as the right next step if this
setup ever needs to be automated rather than run by hand.

## 6. The one mental-model shift that actually matters coming from AWS

Don't think:

> "I need one access key for my entire GCP organization."

Think instead:

> "I create identities, and give each one the minimum IAM permissions it needs, at
> the appropriate level of the resource hierarchy."

```
Organization
│
├── Folder: Dev
│     ├── Project: App-Dev    → Service Account: app-dev-sa
│     └── Project: ML-Dev     → Service Account: ml-dev-sa
│
└── Folder: Prod
       ├── Project: App-Prod  → Service Account: app-prod-sa
       └── Project: ML-Prod   → Service Account: ml-prod-sa
```

Because IAM inherits down the hierarchy, a grant at the Folder level automatically
applies to every Project underneath it — the lever AWS reaches for with an SCP at the
OU level, GCP reaches for with an IAM binding at the Folder level.

**In this repo:** one Project, one narrowly-scoped Service Account
(`${var.project}-vm`), created fresh by this module rather than reused from anywhere
else — the smallest possible instance of this same pattern, not an exception to it.

## 7. The `gcloud` CLI

| Purpose | AWS | GCP |
|---|---|---|
| Main CLI | `aws` | `gcloud` |
| Object storage | `aws s3` | `gcloud storage` |
| Kubernetes | `aws eks` / `kubectl` | `gcloud container` + `kubectl` |
| Terraform | `terraform` | `terraform` (unchanged) |
| Authenticate (human) | `aws configure` | `gcloud auth login` |
| Authenticate (apps/Terraform) | env vars / `~/.aws/credentials` | `gcloud auth application-default login` |
| Set active project | n/a (per-profile) | `gcloud config set project` |
| List projects | `aws organizations ...` | `gcloud projects list` |
| IAM | `aws iam` | `gcloud iam` |
| Service accounts | `aws iam` (roles) | `gcloud iam service-accounts` |

`gcloud storage` is the interface to learn first — `gsutil` still exists and still
works, but `gcloud storage` is Google's current recommended path.

### The basic workflow

```
gcloud auth login                        # human login, once
       ↓
gcloud config set project PROJECT_ID     # every later command defaults to this project
       ↓
gcloud <service> <command>               # e.g. gcloud compute instances list
```

Note the **second**, separate auth step Terraform (and any app/SDK code) actually
needs:

```bash
gcloud auth application-default login
```

`gcloud auth login` authenticates *you*, for `gcloud` commands. Application Default
Credentials are a *separate* login that Terraform, client libraries, and this
module's own `google` provider block all read — both are listed in this module's
[README prerequisites](../README.md#prerequisites-one-time-on-the-mac) because it's
an easy step to do one of and assume you're done.

### Commands worth memorizing first

```bash
gcloud auth login
gcloud auth list
gcloud config list
gcloud config set project PROJECT_ID
gcloud projects list
gcloud compute instances list
gcloud storage ls
gcloud iam service-accounts list
gcloud container clusters list
gcloud projects get-iam-policy PROJECT_ID
```

### IAM from the CLI — the exact shape `iam.tf` automates

```bash
# List service accounts in the active project
gcloud iam service-accounts list

# Create one by hand (this module does this via google_service_account instead)
gcloud iam service-accounts create my-app-sa

# Grant it a role, scoped to the whole project (this module scopes to a single
# bucket/instance instead — see section 3 above for why that's tighter)
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:my-app-sa@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"
```

### Compute — this module's actual object

```bash
gcloud compute instances list
gcloud compute instances describe INSTANCE_NAME --zone=ZONE   # what `make status` runs
gcloud compute instances stop INSTANCE_NAME --zone=ZONE       # what `make stop` runs
```

### GKE (not used by this module, included for completeness given a Kubernetes/EKS background)

```bash
gcloud container clusters list
gcloud container clusters get-credentials my-cluster --region us-central1
kubectl get nodes
```

```
gcloud
  │
  └── GCP infrastructure / GKE cluster management
              │
              ↓
           kubectl
              │
              ↓
       Kubernetes resources (once credentials are fetched)
```

Same two-layer shape as `aws eks update-kubeconfig` → `kubectl` — `gcloud`/`aws`
manage the cluster and hand `kubectl` its credentials; `kubectl` itself is identical
either way, since it only ever speaks the Kubernetes API, not a cloud-specific one.

### Drilling into help, rather than memorizing every flag

```bash
gcloud help
gcloud compute help
gcloud compute instances help
gcloud iam help
gcloud container help
```
