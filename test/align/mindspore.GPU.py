# import mindspore
# from mindspore.communication import get_rank, get_group_size

# # 初始化mindspore环境（仅在多GPU训练时需要）
# mindspore.communication.init()

# # 获取当前进程（rank）ID
# current_rank = get_rank()

# # 获取总的设备数（包括所有GPU）
# num_devices = get_group_size()

# print(f"当前环境中的总GPU数量是: {num_devices}")

# from mindspore.communication import init, get_rank
# init()
# rank_id = get_rank()
# print(rank_id)


def get_gpu_num():
    import argparse
    parser = argparse.ArgumentParser(description="处理命令行参数")
    parser.add_argument('--num-gpus', type=int, default=1, help='GPU的数量')
    args = parser.parse_args()
    return args.num_gpus

if __name__ == "__main__":
    print(get_gpu_num())