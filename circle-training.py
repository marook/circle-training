#!/usr/bin/python2

import PIL
from pybrain import datasets, structure
from pybrain.supervised import trainers
from pybrain.tools import shortcuts
import random

MAX_TURNS = 1000
MAX_EPOCHS = 100

def main():
    net = shortcuts.buildNetwork(2, 3, 3, 3, 3, 3, 1, hiddenclass=structure.TanhLayer, bias=True)

    for turn in xrange(MAX_TURNS):
        print 'Running turn %s...' % (turn, )
        
        ds = generate_tranining_ds()
        trainer = trainers.BackpropTrainer(net, ds)

        for epoch in xrange(MAX_EPOCHS):
            mse = trainer.train()

        write_net_to_image('activation_%03d.png' % (turn, ), net, ds)

        print 'Last MSE %s' % (mse, )

def generate_tranining_ds(sample_count=1000):
    ds = datasets.SupervisedDataSet(2, 1)

    for i in xrange(sample_count):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        net_in = (x, y)

        r2 = x*x + y*y

        expected_activation = (1.0, ) if r2 <= 1*1 else (-1.0, )

        ds.appendLinked(net_in, expected_activation)

    return ds

def write_net_to_image(image_path, net, ds, image_size=(100, 100)):
    width, height = image_size
    
    img = PIL.Image.new('L', image_size)
    pixels = img.load()

    for y in xrange(height):
        for x in xrange(width):
            net_out = net.activate((x / float(width) * 2.0 - 1.0, y / float(height) * 2.0 - 1.0))

            pixels[x, y] = net_out_to_pixel(net_out[0])

    #for net_in, net_out in ds:
    #    x = min(width, max(0, (net_in[0] + 1.0) / 2.0 * width))
    #    y = min(height, max(0, (net_in[1] + 1.0) / 2.0 * width))
    #
    #    pixels[x, y] = net_out_to_pixel(net_out[0])

    img.save(image_path, 'PNG')

def net_out_to_pixel(net_out):
    return min(255, max(0, int((net_out + 1.0) / 2.0 * 255)))

if __name__ == '__main__':
    main()
