# We turn most of the program into a function since we need to call it from our main.py file
# Import some built-in python libraries
import os
import numpy

# Import libraries to scrape youtube
import scrapetube
import youtube_dl


class Audio:

  def __init__(self, artist, artist_channel_link, parent_path) -> None:
    self.artist = artist.replace(' ', '_')
    self.artist_channel_link = artist_channel_link.replace(
      'https://www.youtube.com/channel/', '')
    self.parent_path = parent_path

  # Let's make sure our audio length is over two minutes
  def check_audio_length(self, full_file_path):
    # Use linux command to check audio length
    command = f'ffprobe -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {full_file_path} 2>/dev/null'
    seconds = os.popen(command).read()
    # Check if audio is above 120 seconds or over two minutes
    if float(seconds) >= 120.0:
      return True
    else:
      return False

  # We have this function as we must use proper headers when using wav files in tensorflow
  # This means, that we can't simply put the .wav extension and have to convert the file properly
  def convert_to_wav(self, input_file):
    if os.path.isfile(input_file):
      new_file = input_file.replace('.mp4', '.wav')
      # This command will convert the file to wav, the ">/dev/null 2>&1" part simply hides the output
      command = f'ffmpeg -i {input_file} -vn -acodec pcm_s16le -ar 44100 -ac 2 {new_file} >/dev/null 2>&1'
      # We're running the command using os.system here
      os.system(command)
      # Here we're removing the old mp4 file
      os.remove(input_file)

      # Make sure audio is over two minutes

      command = f'ffmpeg -i {new_file} -t 120 -c copy {new_file.replace(".wav", "0.wav")}'
      os.system(command)
      os.remove(new_file)
      # Rename the file to the proper name
      os.rename(new_file.replace(".wav", "0.wav"), new_file)
      
      return new_file

  # Let's normalize the audio down to 2 min
  def normalize_audio(self, input_file, last_file):
    if os.path.isfile(input_file):
        # This command will trim the file down to 120 seconds or 2 min, the ">/dev/null 2>&1" part simply hides the output
        command = f'ffmpeg -i {input_file} -ss 0 -to 120 -c copy {last_file} >/dev/null 2>&1'
        os.system(command)
        os.remove(input_file)

  # Let's get our vocals by scraping the youtube channel
  def get_vocals(self):
    counter = 0
    limit = 30

    local_channel = scrapetube.get_channel(self.artist_channel_link)
    base_output_file_path = os.path.join(self.parent_path, self.artist)

    # Let's look through the urls of the videos that the artist's channel has posted
    for vid in local_channel:

      video_id = vid['videoId']
      video_url = f'https://www.youtube.com/watch?v={video_id}'

      # This allows us to take 4 videos (songs) from each artist, seperated by 5 videos each.
      # This DOES fall apart when an artist has less than that many songs, but that's an edge case that I'm not considering right now.

      if (counter % 3 == 0 or counter == 0) and counter <= limit:

        out_file = os.path.join(base_output_file_path,
                                f'{self.artist}-{int(counter/3)}.mp4')

        video_info = youtube_dl.YoutubeDL().extract_info(url=video_url,
                                                         download=False)
        options = {
          'format': 'bestaudio/best',
          'keepvideo': False,
          'outtmpl': out_file,
          #'cachedir': False,
        }
        while True:
          with youtube_dl.YoutubeDL(options) as ydl:
            try:
              ydl.download([video_info['webpage_url']])
              break
            except:
              ydl.cache.remove()
        
        if self.check_audio_length(out_file) == False:
            limit += 3
            counter += 1
            os.remove(out_file)
            continue

        self.convert_to_wav(out_file)

        #out_file = os.path.join(base_output_file_path,
        #                       f'{self.artist}0{int(counter/5)}.wav')
        #last_file = os.path.join(base_output_file_path,
        #                         f'{self.artist}{int(counter/5)}.wav')
        #self.normalize_audio(out_file, last_file)

      counter += 1


def start(PARENT_PATH):
  dump_directory = os.path.join(os.getcwd(), 'audio')
  os.makedirs(dump_directory, exist_ok=True)

  artists = numpy.empty(5, dtype=object)
  artists = ['Kendrick Lamar', 'Ariana Grande', 'Travis Scott']

  artist_channels = numpy.empty(5, dtype=object)
  artist_channels = [
    'https://www.youtube.com/channel/UC3lBXcrKFnFAFkfVk5WuKcQ',
    'https://www.youtube.com/channel/UC9CoOnJkIBMdeijd9qYoT_g',
    'https://www.youtube.com/channel/UCtxdfwb9wfkoGocVUAJ-Bmg'
  ]

  for i in range(numpy.size(artists)):
    artist = Audio(artists[i], artist_channels[i], PARENT_PATH)
    print(f'Getting Clip {i}')
    artist.get_vocals()


if __name__ == "__main__":

  directory = os.path.join(os.getcwd(), 'audio')
  start(directory)


  def normalize_audio(input_file, last_file):
    if os.path.isfile(input_file):
        # This command will trim the file down to 120 seconds or 2 min, the ">/dev/null 2>&1" part simply hides the output
        command = f'ffmpeg -i {input_file} -ss 0 -to 120 -c copy {last_file} >/dev/null 2>&1'
        os.system(command)
        os.remove(input_file)


  
  for dir in os.listdir(directory):
    for filename in os.listdir(os.path.join(directory, dir)):
        f = os.path.join(directory, dir, filename)
        last_file = f.replace('-','') 
        normalize_audio(f, last_file)
