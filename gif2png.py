import os
import Image

samples = os.listdir('train')

for i in range(0, len(samples)):
    path = os.path.join('train', samples[i])
    savepath = path[:-3] + 'png'

    im = Image.open(path)

    def iter_frames(im):
        try:
            i = 0
            while 1:
                im.seek(i)
                imframe = im.copy()
                if i == 0:
                    palette = imframe.getpalette()
                else:
                    imframe.putpalette(palette)
                yield imframe
                i += 1
        except EOFError:
            pass


    for i, frame in enumerate(iter_frames(im)):
        frame.save(savepath, **frame.info)
