#!/usr/bin/python2

import PIL
from pybrain import structure
from pybrain.tools import shortcuts

MAX_EPOCHS = 1

def main():
    net = shortcuts.buildNetwork(2, 3, 1, hiddenclass=structure.TanhLayer, bias=False)

    for epoch in xrange(MAX_EPOCHS):
        write_net_to_image('activation_%s.png' % (epoch, ), net)

def write_net_to_image(image_path, net, image_size=(100, 100)):
    img = PIL.Image.new('L', image_size)
    pixels = img.load()

    for y in xrange(image_size[1]):
        for x in xrange(image_size[0]):
            net_out = net.activate((x, y))

            pixels[x, y] = net_out_to_pixel(net_out[0])

    img.save(image_path, 'PNG')

def net_out_to_pixel(net_out):
    return min(255, max(0, int((net_out + 1.0) / 2.0 * 255)))

if __name__ == '__main__':
    main()
