Reminder: the implementation below is based on Amazon EC2 related API.

Dockerfile for scheduler
FROM python:3.7
RUN pip install kubernetes
COPY scheduler.py /scheduler.py
CMD python /scheduler.py

1. Manual setup of Kubernetes cluster

sudo su
apt-get update
apt-get install -y apt-transport-https
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add
cat <<EOF | sudo tee /etc/apt/sources.list.d/kubernetes.list
deb https://apt.kubernetes.io/ kubernetes-xenial main
EOF
apt-get update
apt-get install -y docker.io
apt-get install -y kubelet kubeadm kubectl kubernetes-cni
sudo su
kubeadm init
exit
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
kubectl get node
kubectl get pods --all-namespaces
sudo su
sysctl net.bridge.bridge-nf-call-iptables=1
exit
export kubever=$(kubectl version | base64 | tr -d '\n')
kubectl apply -f "https://cloud.weave.works/k8s/net?k8s-version=$kubever"
kubectl get nodes

2. To use ECR
$(aws ecr get-login --region us-east-1 --no-include-email --registry-ids
763104351884)

3. Distributed Training
# After starting EC2 instances
# Set up inbound rules
eval `ssh-agent`
ssh-add key.pem
# Add to /etc/ssh/ssh_config
Host *
StrictHostKeyChecking no
UserKnownHostsFile=/dev/null
source activate mxnet_p36
# To test
horovodrun -np 4 python examples/horovod/mxnet/train_mxnet_hvd_mnist.py

3. TorchElastic Controller
YAML file config/samples/etcd.yaml
apiVersion: elastic.pytorch.org/v1alpha1
kind: ElasticJob
metadata:
name: imagenet
namespace: elastic-job
spec:
rdzvEndpoint: "etcd-service:2379"
minReplicas: 1
maxReplicas: 3
replicaSpecs:
Worker:
replicas: 2
restartPolicy: ExitCode
template:
apiVersion: v1
kind: Pod
spec:
containers:
- name: elasticjob-worker
image: torchelastic/examples:0.2.0rc1
imagePullPolicy: Always
args:
- "--nproc_per_node=1"
- "/workspace/examples/imagenet/main.py"
- "--arch=resnet18"
- "--epochs=20"
- "--batch-size=32"
- "--workers=0"
- "/workspace/data/tiny-imagenet-200"
resources:
limits:
nvidia.com/gpu: 1


eksctl create cluster --name=torchelastic --region=us-east-1 --ssh-access --sshpublic-
key=~/.ssh/id_rsa.pub --node-type=p2.xlarge --nodes=2 --nodes-min=1 --
nodes-max=3
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/1.0.0-
beta4/nvidia-device-plugin.yml
git clone https://github.com/pytorch/elastic.git
cd elastic/kubernetes
kubectl apply -k config/default
kubectl apply -f config/samples/etcd.yaml
kubectl apply -f config/samples/imagenet.yaml
# Check workers are running
kubectl get pods -n elastic-job
# To simulate disruption get nodegroup name
eksctl get nodegroup --cluster torchelastic
eksctl scale nodegroup --cluster=torchelastic --nodes=1 --name=ng-a345f4e1

4. Kube-scheduler
#!/usr/bin/env python
import time
import random
import json
from kubernetes import client, config, watch
config.load_kube_config()
v1=client.CoreV1Api()
scheduler_name = "random"
def nodes_available():
ready_nodes = []
for n in v1.list_node().items:
for status in n.status.conditions:
if status.status == "True" and status.type == "Ready":
ready_nodes.append(n.metadata.name)
return ready_nodes
def scheduler(name, node, namespace="default"):
body=client.V1Binding()
target=client.V1ObjectReference()
target.kind="Node"
target.apiVersion="v1"
target.name= node
meta=client.V1ObjectMeta()
meta.name=name
body.target=target
body.metadata=meta
return v1.create_namespaced_binding_binding(name, namespace, body)
def main():
w = watch.Watch()
for event in w.stream(v1.list_namespaced_pod, "default"):
if event['object'].status.phase == "Pending" and
event['object'].spec.scheduler_name == scheduler_name:
try:
res = scheduler(event['object'].metadata.name,
random.choice(nodes_available()))
except client.rest.ApiException as e:
print json.loads(e.body)['message']
if __name__ == '__main__':
main()

5. Run ByteScheduler source code
# Repeat for number of nodes necessary
sudo apt install nvidia-340
sudo reboot
git clone -b bytescheduler https://github.com/bytedance/byteps.git>
# On main node
rsync -a ~/byteps ubuntu@remote:/byteps
rsync -a ~/.ssh ubuntu@remote:/.ssh
cd docker
./build-docker-image.sh
# On main node
nvidia-docker run -it --network=host -v ~/.ssh image_name
# Inside container
# Example for 16 GPU across 4 nodes on port 12345
# pytorch_horovod_benchmark.py will be replaced with MLBench
horovodrun -np 16 -H host1:4,host2:4,host3:4,host4:4 -p 12345 python
pytorch_horovod_benchmark.py
# On secondary node
nvidia-docker run -it --network=host -v ~/.ssh image_name \
bash -c "/usr/sbin/sshd -p 12345; sleep infinity"

To use ByteScheduler
model = getattr(models, args.model)(num_classes=args.num_classes)
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = bsc.ScheduledOptimizer(model, optimizer, args.num_warmup_batches +
args.num_iters * args.num_batches_per_iter)