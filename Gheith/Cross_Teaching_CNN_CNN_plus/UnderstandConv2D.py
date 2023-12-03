import os

script_path = ""
try:
    os.path.dirname(os.path.abspath(__file__))
except NameError:
    for root, dirs, files in os.walk(os.getcwd()):
        # Skip 'data' directory and its subdirectories
        if "Data" in dirs:
            dirs.remove("Data")

        if "mainSegmentationChallenge.ipynb" in files:
            script_path = root
            break

if script_path == "":
    raise FileNotFoundError(
        "There is a problem in the folder structure.\nCONTACT gheith.abinader@icloud.com (514)699-5611"
    )

os.chdir(script_path)

from UnderstandConv2DSetup import *

encoder = UNetEncoderK5()
decoder = UNetDecoderK5()

encoder.train()
decoder.train()

images, label = trainloader.dataset[0]["image"], trainloader.dataset[0]["label"]
images = images.unsqueeze(0)
# Get the output of the first convolutional layer
feature = encoder.forward(images)
conv_output = decoder.forward(feature)
conv_output.shape

# Rearrange dimensions and convert to numpy array
conv_output_image = conv_output.permute(0, 2, 3, 1).detach().numpy()
conv_output_image.shape


def conv_max_layer_plot(nrows, ncols, title, image, figsize=(16, 8), color="gray"):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.suptitle(title)

    for i in range(nrows * ncols):
        image_plot = axs[i // ncols, i % ncols].imshow(image[0, :, :, i], cmap=color)
        axs[i // ncols, i % ncols].axis("off")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(image_plot, cax=cbar_ax)
    plt.show()


def fdl_layer_plot(image, title, figsize=(16, 8), color="gray"):
    fig, axs = plt.subplots(1, figsize=figsize)
    fig.suptitle(title)
    image_plot = axs.imshow(image, cmap=color)
    fig.colorbar(image_plot)
    axs.axis("on")
    plt.show()

def conv_single_image_plot(title, image, figsize=(16, 8), color="gray"):
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title)

    # Display the image
    image_plot = ax.imshow(image[0, :, :, 0], cmap=color)  # Assuming image[0, :, :, 0] is the desired image
    ax.axis("off")

    # Add colorbar
    fig.colorbar(image_plot, ax=ax, fraction=0.046, pad=0.04)

    plt.show()

# conv_max_layer_plot(nrows=1, ncols=1, title="First Conv2D", image=conv_output_image)
conv_single_image_plot(title="First Conv2D", image=conv_output_image)


encoder = UNetEncoderK3()
decoder = UNetDecoderK3()

encoder.train()
decoder.train()

images, label = trainloader.dataset[0]["image"], trainloader.dataset[0]["label"]
images = images.unsqueeze(0)
# Get the output of the first convolutional layer
feature = encoder.forward(images)
conv_output = decoder.forward(feature)

make_dot(conv_output, params=dict(list(encoder.named_parameters())+list(decoder.named_parameters()))).render("rnn_torchviz", format="png")