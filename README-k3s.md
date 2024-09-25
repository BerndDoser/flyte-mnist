# Kubernetes

## List pods

```bash
kubectl get pod -A -o wide
kubectl get pod --namespace=seminar-development -o wide
```

## Print logs

```bash
kubectl logs --namespace=seminar-development < pod_name >
```

## Enter running pod

```bash
kubectl exec --stdin --tty --namespace=seminar-development < pod_name > -- /bin/bash
```
