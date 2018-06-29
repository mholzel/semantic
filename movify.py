import cv2


def process(frame_generator, frame_processor=None, output_path='./movie.mp4', fps=30):
    '''
    This function processes the frames of the frame generator
    with the specified frame processor. The output will be saved as a movie
    to the specified output path with the specified fps.
    '''
    output_video = None

    # Now, for each frame of the video, run the frame processor and
    # save the output in a new video
    fourcc = 0x00000021
    fourcc = 0x7634706d
    for count, frame in enumerate(frame_generator):
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if output_video is None:
            output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
        try:
            if frame_processor:
                frame = frame_processor(frame)
            output_video.write(frame)
        except Exception as ex:
            print("Exception", ex, " when processing frame", count)
            output_video.write(frame)
    cv2.destroyAllWindows()
    if output_video is not None:
        output_video.release()
