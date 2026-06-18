import subprocess
# import multiprocessing as mp
import torch.multiprocessing as mp
import os
import torch
import argparse


def train(pid, command: str, action, gpu_count, gpu_ids):
    command = f"./pipe.sh --action {action} {command} --shard-id {pid} --gpu {gpu_ids[pid]}"
    # print(command)
    printit = True if pid == 0 else False
    result = subprocess.run(["bash", "-c", command], capture_output=(not printit), text=True)
    if result.returncode != 0:
        print(f"[shard {pid}] command failed: {command}")
        if result.stdout:
            print(f"[shard {pid}] stdout:\n{result.stdout}")
        if result.stderr:
            print(f"[shard {pid}] stderr:\n{result.stderr}")
        result.check_returncode()


def main(args):
    gpu_ids = eval(args.gpu+',')
    gpu_count = len(gpu_ids)
    # 特判一下，前10个别用8GPU，eval_lm的count可能==0
    # if '--dataset daily' in args.command:
    #     if '--date 1' in args.command:
    #         gpu_count = 1
    #         gpu_ids = gpu_ids[0:1]
        # for i in range(2, 10):
        #     if f'--date {i}' in args.command:
        #         gpu_count = 4
        #         gpu_ids = gpu_ids[0:4]
    print(f'Use {gpu_count} GPUs, {gpu_ids}')

    mp.spawn(train, args=(args.command, args.action, gpu_count, gpu_ids), nprocs=gpu_count, join=True)
    # process_list = []
    # for i in gpu_ids:
    #     printit = True if i == gpu_ids[0] else False
    #     p = os_context.Process(target=train, args=(is_end, args.command, gpu_count, i, printit))
    #     process_list.append(p)
    # for i in range(gpu_count):
    #     process_list[i].start()
    # for i in range(gpu_count):
    #     process_list[i].join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--command', type=str, default=None)
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--action', type=str, default=None)

    args = parser.parse_args()
    print(args)
    main(args)
