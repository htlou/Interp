import imageio

with imageio.get_writer('stream_plots.gif', mode='I', duration=0.5) as writer:
    for i in range(8, 192, 8):
        stream_path = f"assets/stream_{i}.png"
        image = imageio.imread(stream_path)
        writer.append_data(image)

with imageio.get_writer('layer_plots.gif', mode='I', duration=0.5) as writer:
    for i in range(8, 192, 8):
        layer_path = f"assets/layer_{i}.png"
        image = imageio.imread(layer_path)
        writer.append_data(image)
    