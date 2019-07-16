##################################################################################
#    Gabriel Cano                                                                #
##################################################################################
#    Create a csv file for a provided subset of tags.                            #
#                                                                                #
#                                                                                #
##################################################################################
import pandas as pd



def main(datapath, savepath, tags):
    df = pd.read_csv(datapath)
    dfs = []
    for tag in tags:
        dfs.append(df[df['tag'] == tag])

    newdf = pd.concat(dfs)
    newdf.to_csv(savepath, index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('datapath', type=str,
        help='Full path to the csvfile where the info is.')
    parser.add_argument('savepath', type=str,
        help='Full path to the csvfile where new data will be saved.')
    parser.add_argument('tags', default=[], nargs='+', type=str, 
        help='Specify which tags you want to include.')
    
    args = parser.parse_args()

    # raise exceptions if given illegal arguments
    if '.csv' not in args.datapath:
        raise Exception('datapath must be a csvfile (got {})'.format(args.datapath))
    if '.csv' not in args.savepath:
        raise Exception('savepath must be a csvfile (got {})'.format(args.savepath))

    main(args.datapath, args.savepath, args.tags)
