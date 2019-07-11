##################################################################################
#    Gabriel Cano                                                                #
##################################################################################
#    A script to chop up or pad audio clips to prepare them for a net.           #
#                                                                                #
#                                                                                #
##################################################################################
from scipy.io import wavfile
import csv
from math import floor



def main(datapath, savepath, dur, overlap):
    datafile = open(datapath, 'r')
    savefile = open(savepath, 'w')
    rows = csv.reader(datafile)
    writer = csv.writer(savefile)

    for filepath, tag in rows:
        sr, clip = wavfile.read(filepath)
        splice_size = round(dur * sr)
        stride = round((1 - overlap) * splice_size)

        if splice_size > len(clip):  # if shorter than duration, need padding
            pass

        else:  # if longer chop it
            num_chops = floor((1 + overlap) * len(clip) / splice_size)
            for i in range(num_chops):
                chop_start = i * stride
                chop_end = chop_start + splice_size
                writer.writerow([filepath, tag, chop_start, chop_end])

            writer.writerow([filepath, tag, len(clip) - splice_size, len(clip)])  # add chop at end

    datafile.close()
    savefile.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dur', dest='dur', type=float, 
        help='The time in seconds that each clip should last.')
    parser.add_argument('--overlap', dest='overlap', type=float,
        help='The fraction of overlap, e.g. .5 for 50% overlap.')
    parser.add_argument('--datapath', dest='datapath', type=str,
        help='Full path to the csvfile where the info is.')
    parser.add_argument('--savepath', dest='savepath', type=str,
        help='Full path and filename for where to save new csvfile.')
    args = parser.parse_args()

    main(args.datapath, args.savepath, args.dur, args.overlap)

