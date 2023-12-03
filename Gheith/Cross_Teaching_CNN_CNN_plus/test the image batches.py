import matplotlib.pyplot as plt

# Initialize lists to accumulate images and labels
accumulated_images = []
accumulated_masks = []

for i_batch, sampled_batch in enumerate(trainloader):
    image, label = sampled_batch['image'], sampled_batch['label']
    accumulated_images.append(image)
    accumulated_masks.append(label)

# Flatten the lists of batches into single lists of images and masks
all_images = torch.cat(accumulated_images, dim=0)
all_masks = torch.cat(accumulated_masks, dim=0)


#Display images and masks 8 at a time
for i in range(0, len(all_images), 8):
    plt.figure(figsize=(15, 8))

    for j in range(8):
        if i + j >= len(all_images):  # Check to avoid index error at the end
            break

        # Display image
        plt.subplot(4, 4, 2 * j + 1)
        plt.imshow(all_images[i + j].permute(1, 2, 0))  # assuming image is in (C, H, W) format
        plt.title(f"Image {i + j}")
        plt.axis('off')

        # Display corresponding mask
        if i + j < len(all_masks):  # Check to make sure we have a mask for this image
            plt.subplot(4, 4, 2 * j + 2)
            plt.imshow(all_masks[i + j].permute(1, 2, 0))  # assuming mask is in (C, H, W) format
            plt.title(f"Mask {i + j}")
            plt.axis('off')

    plt.show()