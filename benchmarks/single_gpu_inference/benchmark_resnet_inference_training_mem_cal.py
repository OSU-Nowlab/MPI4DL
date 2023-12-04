import time
import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

# from torchvision.models import resnet50
from torchvision.models import resnet101

# from torchgems.utils import get_gpu_memory

# parser = argparse.ArgumentParser()
# parser.add_argument("--precision", help = "Precision", choices = ["fp_32", "fp_16", "int_8"])
# parser.add_argument("--image-size", help = "Image Size")
# parser.add_argument("--batch-size", help = "Batch Size")
from torchgems import parser

# from models import resnet

parser_obj = parser.get_parser()
args = parser_obj.parse_args()

datapath = args.datapath

batch_size = args.batch_size
parts = 1
image_size = args.image_size
resnet_n = 12
num_classes = args.num_classes
precision = args.precision  # values [fp_32, fp_16, bf_16, int_8]
device = "cuda:0"


def get_depth(version, n):
    if version == 1:
        return n * 6 + 2
    elif version == 2:
        return n * 9 + 2


def get_model():
    # model = resnet.get_resnet_v2(
    #     (int(batch_size / parts), 3, image_size, image_size),
    #     depth=get_depth(2, resnet_n),
    #     num_classes=num_classes,
    # )
    model = resnet101()
    return model


def load_dataset():
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    testset = torchvision.datasets.FakeData(
        size=10 * batch_size,
        image_size=(3, image_size, image_size),
        num_classes=num_classes,
        transform=transform,
        target_transform=None,
        random_offset=0,
    )
    dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return dataloader


def evaluate(dataloader, model):
    corrects = 0
    t = time.time()
    perf = []
    count = 0
    size_dataset = len(dataloader.dataset)
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            if count > math.floor(size_dataset / (batch_size)) - 1:
                break
            count += 1
            inputs, labels = data
            if precision == "fp_16":
                inputs, labels = inputs.to(device, dtype=torch.float16), labels.to(
                    device, dtype=torch.float16
                )
            elif precision == "bfp_16":
                assert (
                    torch.cuda.is_bf16_supported() == True
                ), "Native System doen't support bf16"
                inputs, labels = inputs.to(device, dtype=torch.bfloat16), labels.to(
                    device, dtype=torch.bfloat16
                )
            else:
                inputs, labels = inputs.to(device), labels.to(device)

            start_event = torch.cuda.Event(enable_timing=True, blocking=True)
            end_event = torch.cuda.Event(enable_timing=True, blocking=True)
            start_event.record()

            outputs = model(inputs)

            end_event.record()
            torch.cuda.synchronize()
            t = start_event.elapsed_time(end_event) / 1000
            perf.append(len(inputs) / t)

            _, predicted = torch.max(outputs, 1)
            corrects += (predicted == labels).sum().item()
    accuracy = corrects / len(dataloader.dataset)
    print(f"Mean {sum(perf) / len(perf)} Median {np.median(perf)}")
    print(f"Accuracy : {accuracy}")
    return accuracy


def evaluate_train(dataloader, model):
    corrects = 0
    t = time.time()
    perf = []
    count = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    size_dataset = len(dataloader.dataset)
    for batch, data in enumerate(dataloader):
        if count > math.floor(size_dataset / (batch_size)) - 1:
            break
        count += 1
        inputs, labels = data
        if precision == "fp_16":
            inputs, labels = inputs.to(device, dtype=torch.float16), labels.to(
                device, dtype=torch.float16
            )
        elif precision == "bfp_16":
            assert (
                torch.cuda.is_bf16_supported() == True
            ), "Native System doen't support bf16"
            inputs, labels = inputs.to(device, dtype=torch.bfloat16), labels.to(
                device, dtype=torch.bfloat16
            )
        else:
            inputs, labels = inputs.to(device), labels.to(device)

        start_event = torch.cuda.Event(enable_timing=True, blocking=True)
        end_event = torch.cuda.Event(enable_timing=True, blocking=True)
        start_event.record()

        outputs = model(inputs)

        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        end_event.record()
        torch.cuda.synchronize()
        t = start_event.elapsed_time(end_event) / 1000
        perf.append(len(inputs) / t)

        _, predicted = torch.max(outputs, 1)
        corrects += (predicted == labels).sum().item()

    accuracy = corrects / len(dataloader.dataset)
    print(f"Mean {sum(perf) / len(perf)} Median {np.median(perf)}")
    print(f"Accuracy : {accuracy}")
    return accuracy


def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"model size: {size_all_mb:.3f} MB")
    print(f"Number of Params : {sum(p.numel() for p in model.parameters())}")


def run_inference():
    print(
        f"\n\n*********** image_size : {image_size} Precision : {precision}  batch_size : {batch_size} num_classes : {num_classes}  ***********"
    )
    # get_gpu_memory(rank=0)
    max_memory_before = torch.cuda.max_memory_allocated(device="cuda")

    model = get_model().to(device)
    print(
        f'after model load : {torch.cuda.max_memory_allocated(device="cuda") / (1024 ** 2):.2f} MB'
    )
    dataloader = load_dataset()
    accuracy = evaluate(dataloader, model)
    print(f"Accuracy with pretrained model : {accuracy * 100}")
    print_model_size(model)
    max_memory_after = torch.cuda.max_memory_allocated(device="cuda")

    print(
        f"Max Memory Before  Using PyTorch CUDA: {max_memory_before / (1024 ** 2):.2f} MB"
    )
    print(
        f"Max Memory After Using PyTorch CUDA: {max_memory_after / (1024 ** 2):.2f} MB"
    )


run_inference()


# def run_training():
#     print(
#         f"\n\n*********** image_size : {image_size} Precision : {precision}  batch_size : {batch_size} num_classes : {num_classes}  ***********"
#     )
#     # get_gpu_memory(rank=0)
#     max_memory_before = torch.cuda.max_memory_allocated(device="cuda")

#     model = get_model().to(device)
#     print(
#         f'after model load : {torch.cuda.max_memory_allocated(device="cuda") / (1024 ** 2):.2f} MB'
#     )
#     dataloader = load_dataset()
#     accuracy = evaluate_train(dataloader, model)
#     print(f"Accuracy with pretrained model : {accuracy * 100}")
#     print_model_size(model)
#     max_memory_after = torch.cuda.max_memory_allocated(device="cuda")

#     print(
#         f"Max Memory Before  Using PyTorch CUDA: {max_memory_before / (1024 ** 2):.2f} MB"
#     )
#     print(
#         f"Max Memory After Using PyTorch CUDA: {max_memory_after / (1024 ** 2):.2f} MB"
#     )


# run_training()
