# python3 firststep_crop_video.py --inp ../notebooks/de_capture2.mp4 \
#                       --min_frames 15 \
#                       --outputcroppedvideo crop_video.mp4 \
#                       --cpu


# python3 secondstep_synthesize_video.py  --config ../config/vox-256.yaml \
#                 --driving_video ./crop_video.mp4 \
#                 --source_image /home/vuthede/Desktop/face2.png \
#                 --checkpoint ~/Downloads/vox-cpk.pth.tar \
#                 --relative \
#                 --adapt_scale \
#                 --cpu\
#                 --find_best_frame \
#                 --result_video synthesis_video.mp4

python3 thirdstep_align_video.py --synthesisvideo synthesis_video.mp4 \
                                 --fullhdvideo ../obama_fullhd.mp4 \
                                 --outputvideo align_result.mp4   