import pylab
import os
import re
import argparse

from glob import glob

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Arguments for plotting losses")

    parser.add_argument("--dir", type=str, \
            help="directory containing logs", default="./log/")
    
    args = parser.parse_args()
    
    files = sorted(glob(os.path.join(args.dir, '*.log')))

    for f in files:

        print(f)
        name = os.path.basename(f)[:-4]#f.split('/')[-1]
        train_generator_loss = list()
        train_discriminator_loss = list()

        r = open(f,'r')
        for line in r:
            try:
                if "Train Generator" in line:
                    n = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    train_generator_loss.append(float(n[0]))
                if "Train Discriminator" in line:
                    n = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    train_discriminator_loss.append(float(n[0]))
            except Exception as e:
                pass

        r.close()
        pylab.figure(figsize=(12,12))
        pylab.plot(train_generator_loss, 'r', label="generator loss")
        pylab.plot(train_discriminator_loss, 'g', label="discriminator loss")
        pylab.grid(), pylab.legend(loc=1)
        pylab.title(name)
        pylab.savefig(args.dir+name+'.png')
        pylab.close()
