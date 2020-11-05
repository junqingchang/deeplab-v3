import torch
from voc import VOCSegmentation
import random
import matplotlib.pyplot as plt
from PIL import Image



voc_checkpoint_dir = 'chkpt/voc-epoch100.pt'

if __name__ == '__main__':
    model = torch.load(voc_checkpoint_dir)
    model.to('cpu').eval()
    val_data = VOCSegmentation('data/', image_set='val', h=None, w=None)

    sample_data = val_data[random.randint(0, len(val_data)-1)]
    sample = sample_data[0]
    sample_output = sample_data[1]
    with torch.no_grad():
        output = model(sample.unsqueeze(0))[0]
    output_predictions = output.argmax(0)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy())
    r.putpalette(colors)

    s = Image.fromarray(sample_output.byte().cpu().numpy())
    s.putpalette(colors)

    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(sample.transpose(0, 1).transpose(1, 2).long())
    plt.subplot(1, 3, 2)
    plt.title('Prediction')
    plt.imshow(r)
    plt.subplot(1, 3, 3)
    plt.title('Target')
    plt.imshow(s)
    plt.show()