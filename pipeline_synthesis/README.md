# Pipeline to Crop guiding video, synthesize and align to the fullhd video

- Edit parameter in the `./pipeline.sh`. Some parameters need to be define:
 + `--inp`: Input of guiding video need to be cropped
 + `--outputcroppedvideo`: output of cropped video

 + `--driving_video`: The guiding video. It should be the same as `--outputcroppedvideo`
 + `--result_video`: The output synthesis video

 + `--synthesisvideo` The synthesis video
 + `--fullhdvideo` The fullhd video
 + `--outputvideo` The final output