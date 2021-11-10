import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

img_nb = 800
image_path = 'imgs_rot/{}.png'.format(img_nb)
label_path = 'labels_rot/{}.txt'.format(img_nb)

im = Image.open(image_path)
res_x, res_y = im.size
labels_list = (
        open(label_path, 'r')
        .readlines()
        )
labels_list = map(lambda line : line[2:].split(' '), labels_list)

# Create figure and axes
fig, ax = plt.subplots()

# Display the image
ax.imshow(im)

for labels in labels_list :

    center_x, center_y, width, height  = float(labels[0])*res_x, float(labels[1])*res_y, float(labels[2])*res_x, float(labels[3])*res_y
    #correction to place center to center
    center_x -= width/2
    center_y -= height/2

    # Create a Rectangle patch
    rect = patches.Rectangle( (center_x, center_y), width, height, linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)

plt.show()

'''
0 0.23522142321636094 0.7351855354892481 0.3155394345499443 0.35625072425000215
1 0.46882647778066433 0.6522805235038418 0.07569996739076806 0.10546695547238771
2 0.8660376952646771 0.8002274206335905 0.2531543471673452 0.3723730880519771
3 0.5895443565997134 0.8859341339512101 0.16910577838621665 0.22813173209757975
'''
