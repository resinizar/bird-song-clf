import sounddevice as sd
import soundfile as sf 
import pandas as pd 
from time import sleep



df = pd.read_csv('./annos/data500-500-1.csv')
df = df.loc[df['tag'] == 'cotr']

for i, row in df.iterrows():
	print(row)
	fp, start, end, sr, tag = row
	data, sr = sf.read(fp, start=int(start), stop=int(end))
	dur_in_sec = len(data) / sr
	print('dur: ', dur_in_sec)
	print()
	sd.play(data, sr)
	sleep(dur_in_sec)  
