# %% [markdown]
# # Kubernetes
# Interactive shell commands to interact with a kubernetes cluster.

# %% [markdown]
# Define the namespace
# %%
namespace="flytesnacks-development"

# List pods
# %%
!kubectl get pod -A -o wide

# %%
!kubectl get pod --namespace={namespace} -o wide


## Print logs of a single pod
# %%
pod_name="anlzzfv2jss8chz6lg4s-n0-0"

# %%
!kubectl logs --namespace={namespace} {pod_name}


## Enter running pod
# %%
!kubectl exec --stdin --tty --namespace={namespace} {pod_name} -- /bin/bash
