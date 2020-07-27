import subprocess

command= 'ffmpeg -i ../notebooks/de_capture2.mp4 -ss 0.0 -t 7.1 -filter:v crop=374:374:172:84,scale=256:256 crop_video.mp4'

# command = ['ffmpeg', '-i', '../notebooks/de_capture2.mp4','-ss', '0.0', '-t', '7.1', '-filter:v', 'crop=374:374:172:84, scale=256:256', 'crop_video.mp4']

command = command.split(" ")
subprocess.call(command)

