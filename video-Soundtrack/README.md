# Video2Music

🎵 Auto-generate music for your videos!🎵

Video2Music is a novel AI-powered tool built with Flask that can generate music for any short video of max 30 seconds. It is hosted and deployed on Replit, making it easy for anyone to use.

Video2Music works by first extracting images from the video every 2 seconds. It then uses InstructBLIP to identify what is happening in each image. Finally, it uses the Google Text-Bison model (Replit ModelFarm) to generate an overall description of the video.

This information is then used to prompt the same model to generate a music prompt of instruments, genre, mood, feel, time signature, BPM, kbps, and kHz suitable for the video. This prompt is then used to generate music for the entire video length using the MusicGen Facebook model.

Video2Music also includes functionality to alter the video description and music prompt generated so that the user can generate the music to their liking.

Note: The audio cannot be used for commercial purposes since the MusicGen model has weights licensed as CC-BY-NC 4.0 (NC = non-commercial)

![v2m (1)](https://github.com/sam9111/Video2Music/assets/60708693/6a147ea8-6e5d-4860-9741-0c46b3977f14)




https://github.com/sam9111/Video2Music/assets/60708693/131277ef-a80e-4eb9-bfa2-eb5f9e261548

