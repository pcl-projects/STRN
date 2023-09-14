#! /usr/bin/env python3


from zeyu_utils import net as znet

cmd_listener_port = 30009
cmd_client_max_num = 3
rpc_master_addr = "udc-ba26-24-ic.hpc.virginia.edu"
rpc_master_addr = "172.31.92.17"


# def run_remote_py(sockm, pyfname="training_sgd.py", path_prefix="/u/qxc4fh/zeyu_workspace/sgd/", **kwargs):
# def run_remote_py(sockm, pyfname="local_update.py", path_prefix="/home/qxc4fh/zeyu_workspace/sgd/", **kwargs):
def run_remote_py(sockm, pyfname="training_sgd.py", path_prefix="/home/ubuntu/sgd/", **kwargs):
    job_name = kwargs["job_name"]
    model_name = kwargs["model_name"]
    rpc_rank = kwargs["rpc_rank"]
    ps_num = kwargs["ps_num"]
    worker_num = kwargs["worker_num"]
    rpc_master_addr = kwargs["rpc_master_addr"]
    rpc_master_port = kwargs["rpc_master_port"]
    epoch_num = kwargs["epoch_num"]
    data_partitioned = kwargs["data_partitioned"]
    gpu_ids = kwargs["gpu_ids"]

    cmd = f"python3 {path_prefix}/{pyfname} --job_name={job_name} --model_name={model_name} --rpc_rank={rpc_rank} --ps_num={ps_num} --worker_num={worker_num} --rpc_master_addr={rpc_master_addr} --rpc_master_port={rpc_master_port} --epoch_num={epoch_num} --data_partitioned={data_partitioned} --gpu_ids={gpu_ids}"

    sockm.send(cmd)


if __name__ == "__main__":
    client_sockms = []
    cmd_listener = znet.SocketMsger.tcp_listener("0.0.0.0", cmd_listener_port)
    for _ in range(cmd_client_max_num):
        client_sockm, _ = cmd_listener.accept()
        client_sockms.append(client_sockm)

    job_name = "job21"
    model_name = "lstm"
    ps_num = 1
    worker_num = 7
    gpu_ids = "0,1,0,1,0,1,2,2"
    client_sockm_ids = [0, 0, 0, 0, 1, 1, 2, 2, 2, 0]

    for i in range(ps_num + worker_num + 2):
        run_remote_py(
            client_sockms[client_sockm_ids[i]],
            job_name=job_name,
            model_name=model_name,
            rpc_rank=i,
            ps_num=ps_num,
            worker_num=worker_num,
            rpc_master_addr="172.31.92.17",
            rpc_master_port=59670,
            epoch_num=10000000000,
            data_partitioned=1,
            gpu_ids=gpu_ids,
            # model_fetch_ids=model_fetch_ids,
        )

    job_name = "job22"
    model_name = "densenet121"
    ps_num = 1
    worker_num = 7
    gpu_ids = "2,3,3,4,3,4,5,4"
    client_sockm_ids = [1, 1, 1, 1, 2, 2, 0, 0, 0, 1]

    for i in range(ps_num + worker_num + 2):
        run_remote_py(
            client_sockms[client_sockm_ids[i]],
            job_name=job_name,
            model_name=model_name,
            rpc_rank=i,
            ps_num=ps_num,
            worker_num=worker_num,
            rpc_master_addr="172.31.89.15",
            rpc_master_port=59671,
            epoch_num=10000000000,
            data_partitioned=1,
            gpu_ids=gpu_ids,
            # model_fetch_ids=model_fetch_ids,
        )

    job_name = "job23"
    model_name = "gpt"
    ps_num = 1
    worker_num = 7
    gpu_ids = "5,6,6,7,5,6,7,7"
    client_sockm_ids = [2, 2, 2, 2, 0, 0, 1, 1, 1, 2]

    for i in range(ps_num + worker_num + 2):
        run_remote_py(
            client_sockms[client_sockm_ids[i]],
            job_name=job_name,
            model_name=model_name,
            rpc_rank=i,
            ps_num=ps_num,
            worker_num=worker_num,
            rpc_master_addr="172.31.85.164",
            rpc_master_port=59672,
            epoch_num=10000000000,
            data_partitioned=1,
            gpu_ids=gpu_ids,
            # model_fetch_ids=model_fetch_ids,
        )
