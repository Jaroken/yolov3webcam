from moviepy.editor import VideoFileClip, concatenate_videoclips
clip1 = VideoFileClip("C:\\Users\\PRESTK\\Documents\\yolov3test1.avi")
clip2 = VideoFileClip("C:\\Users\\PRESTK\\Documents\\yolov3test2.avi")
clip3 = VideoFileClip("C:\\Users\\PRESTK\\Documents\\yolov3test3.avi")
clip4 = VideoFileClip("C:\\Users\\PRESTK\\Documents\\yolov3test4.avi")
clip5 = VideoFileClip("C:\\Users\\PRESTK\\Documents\\yolov3test5.avi")

final_clip = concatenate_videoclips([clip2,clip3,clip4,clip5])
final_clip.write_videofile("C:\\Users\\PRESTK\\Documents\\my_concatenation.mp4")
