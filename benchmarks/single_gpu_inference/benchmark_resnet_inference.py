import time
import math
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchgems import parser
from torchgems.utils import get_depth

parser_obj = parser.get_parser()
args = parser_obj.parse_args()

datapath = args.datapath
device = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = args.checkpoint
APP = args.app
MPI4DL = False  # Enable if you want to use ResNet designed by MPI4DL

batch_size = args.batch_size
parts = 1
image_size = args.image_size
resnet_n = 12
num_classes = args.num_classes
precision = args.precision  # values [fp_32, fp_16, bf_16, int_8]


def load_custom_dataset(batch_size, image_size):
    num_workers = 1
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    torch.manual_seed(0)

    testset = torchvision.datasets.ImageFolder(
        root=datapath,
        transform=transform,
        target_transform=None,
    )

    dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


def load_fake_dataset(batch_size, image_size):
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


def load_cifar10Test(batch_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    testset = torchvision.datasets.CIFAR10(
        root="./cifar10_dataset", train=False, download=True, transform=transform
    )

    dataloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return dataloader


def load_dataset(app, batch_size, image_size):
    if app == 1:
        return load_custom_dataset(batch_size, image_size)
    if app == 2:
        return load_fake_dataset(batch_size, image_size)
    if app == 3:
        return load_cifar10Test(batch_size)


def int8_quantization_with_tensorRT(
    model, device, batch_size, image_size, num_classes, precision
):
    import torch_tensorrt

    saved_checkpt = f"int8_quant_model_{image_size}_{num_classes}_{batch_size}.pth"

    if os.path.exists(saved_checkpt):
        print(f"Using saved checkpoint at : {saved_checkpt}")
        model = torch.jit.load(saved_checkpt)
        return model
    else:
        model.eval()
        testing_dataloader = load_fake_dataset(batch_size, image_size)
        # important step o caliberation
        # testing_dataloader = load_custom_dataset(batch_size, image_size)
        calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
            testing_dataloader,
            cache_file="./calibration.cache",
            use_cache=False,
            algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            device=torch.device(device),
        )

        print(f"precision : {precision}")
        trt_mod = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input((batch_size, 3, image_size, image_size))],
            enabled_precisions={torch.int8},
            calibrator=calibrator,
            device={
                "device_type": torch_tensorrt.DeviceType.GPU,
                "gpu_id": 0,
                "dla_core": 0,
                "allow_gpu_fallback": False,
                "disable_tf32": False,
            },
        )
        # torch.jit.save(torch.jit.script(trt_mod), saved_checkpt)
        # torch.save(trt_mod.state_dict(),'res_torch_quant_model_stats.pth')
        # trt_mod.to(device)
        return trt_mod


def load_model(
    device, batch_size, parts, image_size, resnet_n, num_classes, MPI4DL, precision
):
    if MPI4DL:
        from models import resnet

        model = resnet.get_resnet_v2(
            (int(batch_size / parts), 3, image_size, image_size),
            depth=get_depth(2, resnet_n),
            num_classes=num_classes,
        )
    else:
        from torchvision.models import resnet50

        model = resnet50(num_classes=num_classes)

    if CHECKPOINT_PATH is not None:
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    model.eval()
    if precision == "fp_16":
        model.half()
    if precision == "bfp_16":
        assert (
            torch.cuda.is_bf16_supported() == True
        ), "Native System doen't support bf16"
        model = model.to(torch.bfloat16)
    elif precision == "int_8":
        model = int8_quantization_with_tensorRT(
            model, device, batch_size, image_size, num_classes, precision
        )
    model.to(device)
    return model


def evaluate(device, model, dataloader, precision):
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


def run_inference(image_size):
    print(
        f"\n\n*********** image_size : {image_size} Precision : {precision}  batch_size : {batch_size} num_classes : {num_classes} APP : {APP} ***********"
    )
    dataloader = load_dataset(app=APP, batch_size=batch_size, image_size=image_size)

    model = load_model(
        device,
        batch_size=batch_size,
        parts=parts,
        image_size=image_size,
        resnet_n=resnet_n,
        num_classes=num_classes,
        MPI4DL=MPI4DL,
        precision=precision,
    )

    accuracy = evaluate(device, model, dataloader, precision)
    print(f"Accuracy with pretrained model : {accuracy * 100}")


run_inference(image_size=image_size)
