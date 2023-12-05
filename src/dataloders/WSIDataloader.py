import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class WSIDataloader(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        # self.tile_size = tile_size
        self.transform = transform
        self.class_list = os.listdir(root)
        self.image_list = []
        self.classes_map = {}

        count = 0
        for class_name in self.class_list:
            self.classes_map[class_name] = count
            count += 1
            class_path = os.path.join(root, class_name)
            if os.path.isdir(class_path):
                image_names = os.listdir(class_path)
                image_paths = [os.path.join(class_path, img) for img in image_names]
                self.image_list.extend(image_paths)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        # wsi_image = Image.open(img_path)
        # torch.tensor(wsi_image).permute(2, 0, 1).contiguous()

        # tiles = self.generate_tiles(wsi_image)
        # tiles = torch.stack(tiles)
        # return path of image and class label
        class_name = os.path.basename(os.path.dirname(img_path))

        return img_path, torch.tensor(self.classes_map[class_name])

    # def generate_tiles(self, image):
    #     tiles = []
    #     width, height = image.size

    #     stride = self.tile_size # check what is the correct way to slide for pathology images. For now, considering sliding completely (i.e. considering independent patches)

    #     for y in range(0, height - self.tile_size + 1, stride):
    #         for x in range(0, width - self.tile_size + 1, stride):
    #             tile = image.crop((x, y, x + self.tile_size, y + self.tile_size))
    #             if self.transform:
    #                 tile = self.transform(tile)
    #             tiles.append(tile)

    #     return tiles


# # Set the path to your main directory containing Class1 and Class2 directories
data_folder = (
    "/home/gulhane.2/GEMS_Inference/datasets/test_digital_pathology/wsi_images"
)

# Define tile size and any other image transformations you need
tile_size = 256  # Adjust the tile size as needed
transform = transforms.Compose(
    [
        transforms.Resize((tile_size, tile_size)),  # Adjust the size as needed
        transforms.ToTensor(),
    ]
)


# ## Utility:
# # Create a custom dataset with tiles
# wsi_dataset = WSIDataloader(root=data_folder, tile_size=tile_size, transform=transform)

# # Create a data loader
# batch_size = 1
# wsi_dataloader = DataLoader(wsi_dataset, batch_size=batch_size, shuffle=True)

# #print(len(wsi_dataloader.dataset)) # no of wsi images
# # Iterate through the data loader in your training loop
# count = 0
# import openslide

# for wsi_imag_path, labels in wsi_dataloader:
#     slide = openslide.OpenSlide(wsi_imag_path[0])
#     tile_size = 1024
#     for x_pix in range(0, slide.dimensions[0], tile_size):
#         for y_pix in range(0, slide.dimensions[1], tile_size):
#             region = slide.read_region((x_pix, y_pix), 0, (tile_size, tile_size))
#             pil_img = region.convert("RGB") # RGBA to RGB

#             transform = transforms.ToTensor()
#             tensor_image = transform(pil_img)

#             tensor_image = tensor_image.unsqueeze(dim=0)
#             print(tensor_image.shape, labels)
#             break
#         break
#     slide.close()


#     count += 1


# print("PYROECH DATALOADER")
# batch_size = 1
# num_workers = 1

# transform = transforms.Compose(
#     [ transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
# )
# torch.manual_seed(0)

# # testset = torchvision.datasets.ImageNet(
# #         root="/home/gulhane.2/GEMS_Inference/datasets/ImageNet/", split='val', transform=transform
# # )
# testset = torchvision.datasets.ImageFolder(
#     root="/home/gulhane.2/github_torch_gems/MPI4DL/benchmarks/single_gpu/imagenette2-320/val",
#     transform=transform,
#     target_transform=None,
# )

# dataloader = torch.utils.data.DataLoader(
#         testset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
# )

# for batch, labels in dataloader:
#     print(batch.shape, labels)
#     break
#     # for tiles in batch:
#     #     print(tiles.shape)
#         # for tile in tiles:
#         #     print(tile.shape)

#     count += 1
