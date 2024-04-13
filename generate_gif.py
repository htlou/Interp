import imageio

# with imageio.get_writer('stream_plots.gif', mode='I', duration=0.5) as writer:
#     for i in range(8, 192, 8):
#         stream_path = f"assets/stream_{i}.png"
#         image = imageio.imread(stream_path)
#         writer.append_data(image)

# with imageio.get_writer('layer_plots.gif', mode='I', duration=0.5) as writer:
#     for i in range(8, 192, 8):
#         layer_path = f"assets/layer_{i}.png"
#         image = imageio.imread(layer_path)
#         writer.append_data(image)

# List of the base names for the different plot types
plot_types = [
    "head", "layer", "stream", "patch_res", "patch_attn", "patch_mlp", "patch_heado", 
    "patch_vdiff", "out_vs_value", "head_pattern", "out_vs_attn"
]

# Loop through each plot type and create a GIF
for plot_type in plot_types:
    gif_filename = f"gif/{plot_type}_plots.gif"  # Name of the gif file
    with imageio.get_writer(gif_filename, mode='I', duration=2) as writer:
        for i in range(8, 192, 8):
            file_path = f"assets/{plot_type}_{i}.png"  # Path to the image file
            image = imageio.imread(file_path)
            writer.append_data(image)
    print(f"GIF saved: {gif_filename}")