python demo.py  --config config/vox-256.yaml \
                --driving_video ./cropyeah.mp4 \
                --source_image /home/vuthede/Desktop/face2.png \
                --checkpoint ~/Downloads/vox-cpk.pth.tar \
                --relative \
                --adapt_scale \
                --cpu\
                --find_best_frame \
                --result_video result2.mp4